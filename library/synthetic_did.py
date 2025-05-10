import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp
from functools import partial, lru_cache
import warnings
warnings.filterwarnings("ignore")


class SyntheticDIDModel:
    def __init__(self, data, metric, period_index, shopno, treated, after_treatment,
                 seed=42, bootstrap_rounds=100, njobs=4, use_parallel=False):
        self.data = data.copy()
        self.outcome_col = metric
        self.period_index_col = period_index
        self.shopno_col = shopno
        self.treat_col = treated
        self.post_col = after_treatment
        self.seed = seed
        self.bootstrap_rounds = bootstrap_rounds
        self.njobs = njobs
        self.use_parallel = use_parallel  # Новый параметр для контроля параллельных вычислений
        # Кэшируем некоторые данные для улучшения производительности
        self._unit_weights = None
        self._time_weights = None
        self._intercept = None
        self._model = None
        self._att = None

    def calculate_regularization(self, data):
        if self.post_col not in data.columns or self.treat_col not in data.columns:
            raise ValueError(f"Отсутствуют необходимые столбцы: {self.post_col} или {self.treat_col}")
            
        n_treated_post = data.loc[(data[self.post_col] == 1) & (data[self.treat_col] == 1)].shape[0]
        first_diff_std = (data
                          .loc[(data[self.post_col] == 0) & (data[self.treat_col] == 0)]
                          .sort_values(self.period_index_col)
                          .groupby(self.shopno_col)[self.outcome_col]
                          .diff()
                          .std())
        return n_treated_post ** (1 / 4) * first_diff_std

    def join_weights(self, data, unit_w, time_w):
        # Оптимизация: проверяем серии весов на None перед соединением
        if unit_w is None or time_w is None or unit_w.empty or time_w.empty:
            # Создаем равномерные веса, если весов нет
            unique_time = pd.unique(data.loc[data[self.post_col] == 1, self.period_index_col])
            unique_units = pd.unique(data.loc[data[self.treat_col] == 1, self.shopno_col])
            
            if len(unique_time) == 0 or len(unique_units) == 0:
                # Если нет данных, возвращаем пустой датафрейм
                return pd.DataFrame(columns=data.columns + ["weights"])
                
            # Создаем равные веса
            if time_w is None or time_w.empty:
                time_w = pd.Series(1.0 / len(unique_time), 
                                  index=unique_time, 
                                  name="time_weights")
            if unit_w is None or unit_w.empty:
                unit_w = pd.Series(1.0 / len(unique_units), 
                                  index=unique_units, 
                                  name="unit_weights")
        
        # Более эффективное соединение без лишних операций
        joined = data.copy()
        
        # Добавляем веса времени
        time_dict = time_w.to_dict()
        joined['time_weights'] = joined[self.period_index_col].map(
            lambda x: time_dict.get(x, 1.0 / len(time_dict) if len(time_dict) > 0 else 1.0)
        )
        
        # Добавляем веса единиц
        unit_dict = unit_w.to_dict()
        joined['unit_weights'] = joined[self.shopno_col].map(
            lambda x: unit_dict.get(x, 1.0 / len(unit_dict) if len(unit_dict) > 0 else 1.0)
        )
        
        # Вычисляем итоговые веса
        joined['weights'] = (joined['time_weights'] * joined['unit_weights'] + 1e-10).round(10)
        
        # Преобразуем столбцы к целым числам
        joined[self.treat_col] = joined[self.treat_col].astype(int)
        joined[self.post_col] = joined[self.post_col].astype(int)
        
        return joined

    def _loss_time_weights(self, w, X, y):
        """Вычисляет среднеквадратичную ошибку для временных весов"""
        return np.sum((w @ X - y) ** 2)

    def _loss_unit_weights(self, w, X, y, zeta, T_pre):
        """Вычисляет среднеквадратичную ошибку с регуляризацией для весов единиц"""
        intercept, weights = w[0], w[1:]
        X_without_intercept = X[:, 1:]
        predicted = intercept + X_without_intercept @ weights
        return np.sum((predicted - y) ** 2) + T_pre * zeta ** 2 * np.sum(weights ** 2)

    def fit_time_weights(self, data):
        control = data.loc[data[self.treat_col] == 0]
        
        # Проверяем, есть ли данные до и после обработки
        control_pre = control.loc[control[self.post_col] == 0]
        control_post = control.loc[control[self.post_col] == 1]
        
        if control_pre.empty or control_post.empty:
            # Возвращаем равные веса, если недостаточно данных
            time_periods = pd.unique(control.loc[control[self.post_col] == 0, self.period_index_col])
            if len(time_periods) == 0:
                time_periods = pd.unique(data.loc[data[self.post_col] == 0, self.period_index_col])
            if len(time_periods) == 0:
                return pd.Series([], name="time_weights", index=[])
            
            return pd.Series(np.ones(len(time_periods)) / len(time_periods),
                           index=time_periods,
                           name="time_weights")
        
        # Оптимизация: используем pivot_table вместо pivot для лучшей обработки дубликатов
        y_pre = (control_pre
                 .pivot_table(index=self.period_index_col, columns=self.shopno_col, 
                              values=self.outcome_col, aggfunc='mean'))
        y_post_mean = (control_post
                       .groupby(self.shopno_col)[self.outcome_col]
                       .mean()
                       .values)
        
        # Проверка наличия достаточных данных
        if y_pre.shape[0] == 0 or y_pre.shape[1] == 0:
            time_periods = pd.unique(control.loc[control[self.post_col] == 0, self.period_index_col])
            return pd.Series(np.ones(len(time_periods)) / len(time_periods),
                           index=time_periods,
                           name="time_weights")
        
        # Сохраняем индекс для последующего использования
        time_index = y_pre.index.copy()
        
        # Формируем матрицу признаков
        X = np.vstack([np.ones(y_pre.shape[1]), y_pre.values])
        
        def objective(w):
            return self._loss_time_weights(w, X, y_post_mean)
        
        # Ограничение: сумма весов = 1 (только для весов времени, без интерсепта)
        def constraint(w):
            return np.sum(w) - 1
        
        # Начальные равные веса
        w_init = np.ones(X.shape[0]) / X.shape[0]
        
        try:
            # Ограничиваем количество итераций для ускорения
            result = fmin_slsqp(
                objective,
                w_init,
                f_eqcons=constraint,
                bounds=[(0.0, None)] * X.shape[0],
                disp=0
            )
            
            # Нормализуем веса
            sum_weights = np.sum(result)
            if sum_weights > 0:
                result_normalized = result / sum_weights
            else:
                result_normalized = np.ones(len(result)) / len(result)
            
            # Важно! Убираем первый вес (интерсепт) и используем оставшиеся для time_weights
            time_weights = pd.Series(result_normalized[1:], name="time_weights", index=time_index)
            
            return time_weights
        except Exception as e:
            # В случае ошибки возвращаем равные веса
            print(f"Ошибка в fit_time_weights: {str(e)}")
            return pd.Series(np.ones(y_pre.shape[0]) / y_pre.shape[0], 
                           name="time_weights", index=time_index)

    def fit_unit_weights(self, data):
        zeta = self.calculate_regularization(data)
        pre_data = data.loc[data[self.post_col] == 0]
        
        # Проверка на наличие данных в предварительном периоде
        pre_control = pre_data.loc[pre_data[self.treat_col] == 0]
        pre_treat = pre_data.loc[pre_data[self.treat_col] == 1]
        
        if pre_control.empty or pre_treat.empty:
            # Возвращаем равные веса и нулевой интерсепт при отсутствии данных
            control_units = pd.unique(data.loc[data[self.treat_col] == 0, self.shopno_col])
            if len(control_units) == 0:
                return pd.Series([], name="unit_weights", index=[]), 0.0
                
            return pd.Series(np.ones(len(control_units)) / len(control_units), 
                          name="unit_weights", index=control_units), 0.0
        
        # Оптимизация: используем pivot_table вместо pivot для лучшей обработки дубликатов
        y_pre_control = (pre_control
                         .pivot_table(index=self.period_index_col, columns=self.shopno_col, 
                                      values=self.outcome_col, aggfunc='mean'))
        y_pre_treat_mean = (pre_treat
                            .groupby(self.period_index_col)[self.outcome_col]
                            .mean())
        
        # Проверка размерностей данных
        if y_pre_control.shape[0] == 0 or y_pre_control.shape[1] == 0 or len(y_pre_treat_mean) == 0:
            control_units = pd.unique(pre_data.loc[pre_data[self.treat_col] == 0, self.shopno_col])
            return pd.Series(np.ones(len(control_units)) / len(control_units), 
                          name="unit_weights", index=control_units), 0.0
        
        T_pre = y_pre_control.shape[0]
        X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)
        y = y_pre_treat_mean.values
        
        def objective(w):
            return self._loss_unit_weights(w, X, y, zeta, T_pre)
        
        # Ограничение: сумма весов = 1, исключая интерсепт
        def constraint(w):
            return np.sum(w[1:]) - 1
        
        # Начальные равные веса + интерсепт 0
        w_init = np.zeros(X.shape[1])
        w_init[1:] = 1.0 / (X.shape[1] - 1)
        
        try:
            # Ограничиваем количество итераций для ускорения
            bounds = [(None, None)] + [(0.0, 1.0)] * (X.shape[1] - 1)
            result = fmin_slsqp(
                objective,
                w_init,
                f_eqcons=constraint,
                bounds=bounds,
                disp=0
            )
            
            # Извлекаем интерсепт и веса
            intercept, weights = result[0], result[1:]
            
            # Нормализуем веса, если нужно
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights_normalized = weights / sum_weights
            else:
                weights_normalized = np.ones(len(weights)) / len(weights)
                
            unit_weights = pd.Series(weights_normalized, name="unit_weights", index=y_pre_control.columns)
            
            return unit_weights, intercept
        except Exception as e:
            # В случае ошибки возвращаем равные веса и нулевой интерсепт
            print(f"Ошибка в fit_unit_weights: {str(e)}")
            return pd.Series(np.ones(y_pre_control.shape[1]) / y_pre_control.shape[1], 
                          name="unit_weights", index=y_pre_control.columns), 0.0

    def synthetic_diff_in_diff(self, data=None, force_recalculate=False):
        # Используем кэшированные результаты, если доступны и не требуется пересчет
        if data is None and not force_recalculate and self._att is not None:
            return self._att, self._unit_weights, self._time_weights, self._model, self._intercept
        
        if data is None:
            data = self.data
            
        try:
            unit_weights, intercept = self.fit_unit_weights(data)
            time_weights = self.fit_time_weights(data)
            
            # Проверка на пустые веса
            if len(unit_weights) == 0 or len(time_weights) == 0:
                return 0.0, unit_weights, time_weights, None, intercept
            
            did_data = self.join_weights(data, unit_weights, time_weights)
            
            # Добавим проверку на достаточность данных
            if did_data.shape[0] == 0:
                return 0.0, unit_weights, time_weights, None, intercept
                
            formula = f"{self.outcome_col} ~ {self.post_col}*{self.treat_col}"
            
            # Не используем веса, равные нулю, чтобы избежать предупреждений в sqrt
            valid_weights = did_data["weights"] > 1e-10
            if valid_weights.sum() < 4:  # Минимальное количество наблюдений для регрессии
                did_model = smf.ols(formula, data=did_data).fit()
            else:
                # Оптимизация: используем только строки с положительными весами
                did_model = smf.wls(formula, data=did_data[valid_weights], 
                                   weights=did_data.loc[valid_weights, "weights"]).fit()
            
            # Проверяем наличие нужного параметра в результатах регрессии
            interaction_term = f"{self.post_col}:{self.treat_col}"
            att = did_model.params.get(interaction_term, 0.0)
            
            # Сохраняем результаты для повторного использования, если это основные данные
            if data is self.data:
                self._att = att
                self._unit_weights = unit_weights
                self._time_weights = time_weights
                self._model = did_model
                self._intercept = intercept
            
            return att, unit_weights, time_weights, did_model, intercept
        except Exception as e:
            print(f"Ошибка в synthetic_diff_in_diff: {str(e)}")
            # Возвращаем нулевые результаты в случае ошибки
            control_units = pd.unique(data.loc[data[self.treat_col] == 0, self.shopno_col])
            time_periods = pd.unique(data.loc[data[self.post_col] == 0, self.period_index_col])
            
            if len(control_units) == 0 or len(time_periods) == 0:
                raise ValueError("Недостаточно данных для создания синтетического контроля")
                
            unit_weights = pd.Series(np.ones(len(control_units)) / len(control_units), 
                                   name="unit_weights", index=control_units)
            time_weights = pd.Series(np.ones(len(time_periods)) / len(time_periods),
                                   name="time_weights", index=time_periods)
            
            return 0.0, unit_weights, time_weights, None, 0.0

    def make_random_placebo(self, data):
        control = data.query(f"~{self.treat_col}")
        shopnos = control[self.shopno_col].unique()
        
        # Проверим, есть ли достаточно магазинов контрольной группы
        if len(shopnos) < 2:
            raise ValueError("Недостаточно контрольных единиц для создания плацебо")
            
        placebo_shopno = np.random.choice(shopnos)
        return control.assign(**{self.treat_col: control[self.shopno_col] == placebo_shopno})

    def _single_placebo_att(self):
        max_attempts = 3  # Уменьшаем количество попыток для ускорения
        
        for attempt in range(max_attempts):
            try:
                # Создаем плацебо-данные с случайной "тестовой" группой из контрольных единиц
                placebo_data = self.make_random_placebo(self.data)
                
                # Быстрые проверки перед расчетами
                treat_units = placebo_data[self.treat_col].sum()
                control_units = (~placebo_data[self.treat_col]).sum()
                
                if treat_units == 0 or control_units == 0:
                    continue
                    
                # Проверка, что у нас есть наблюдения до и после обработки
                pre_obs = (placebo_data[self.post_col] == 0).sum()
                post_obs = (placebo_data[self.post_col] == 1).sum()
                
                if pre_obs == 0 or post_obs == 0:
                    continue
                    
                # Запускаем синтетический diff-in-diff
                att_placebo, *_ = self.synthetic_diff_in_diff(data=placebo_data)
                return att_placebo
                
            except Exception as e:
                if "SVD did not converge" in str(e) and attempt < max_attempts - 1:
                    # Если это ошибка SVD и не последняя попытка, продолжаем
                    print(f"Попытка {attempt+1}: Ошибка в _single_placebo_att: {str(e)}")
                    continue
                else:
                    print(f"Ошибка в _single_placebo_att: {str(e)}")
                    return np.nan
                    
        # Если все попытки не удались
        return np.nan

    def estimate_se(self, alpha=0.05):
        np.random.seed(self.seed)
        try:
            # Используем кэшированный результат, если есть
            if self._att is None:
                main_att, *_ = self.synthetic_diff_in_diff()
            else:
                main_att = self._att
            
            # Если use_parallel=True, используем параллельную обработку
            if self.use_parallel:
                # Оптимизация: увеличиваем batch_size для ускорения параллельных вычислений
                effects = Parallel(n_jobs=self.njobs, batch_size=max(1, self.bootstrap_rounds // (self.njobs * 2)))(
                    delayed(self._single_placebo_att)() for _ in range(self.bootstrap_rounds)
                )
            else:
                # Последовательная обработка (по умолчанию)
                effects = [self._single_placebo_att() for _ in range(self.bootstrap_rounds)]
            
            effects = [e for e in effects if not np.isnan(e)]
            
            if not effects:
                print("Все бутстрэп-итерации завершились с ошибками, возвращаем нулевые значения")
                return 0.0, main_att, main_att  # Возвращаем тот же ATT с нулевым SE и одинаковые границы CI
                
            se = np.std(effects, ddof=1)
            z = norm.ppf(1 - alpha / 2)
            ci_lower = main_att - z * se
            ci_upper = main_att + z * se
            return se, ci_lower, ci_upper 
        except Exception as e:
            print(f"Ошибка в estimate_se: {str(e)}")
            return 0.0, 0.0, 0.0  # Возвращаем нулевые значения в случае ошибки
    
    def rmspe(self):
        """
        Вычисляет Root Mean Squared Prediction Error для контрольной группы
        """
        try:
            # Используем кэшированные значения, если они есть
            if self._att is None:
                _, unit_weights, _, _, intercept = self.synthetic_diff_in_diff()
            else:
                unit_weights, intercept = self._unit_weights, self._intercept
            
            # Получаем данные до воздействия
            pre_data = self.data.loc[self.data[self.post_col] == 0]
            
            # Значения для тестовой группы
            y_treat = pre_data.loc[pre_data[self.treat_col] == 1, [self.period_index_col, self.outcome_col]]
            
            # Если данные отсутствуют, возвращаем 0
            if y_treat.empty:
                return 0.0
            
            # Группируем тестовые данные для эффективности
            y_treat_mean = y_treat.groupby(self.period_index_col)[self.outcome_col].mean()
            
            # Если нет тестовой группы, возвращаем 0
            if len(y_treat_mean) == 0:
                return 0.0
            
            # Значения для контрольной группы - используем pivot_table для эффективности
            y_control = (pre_data
                        .loc[pre_data[self.treat_col] == 0]
                        .pivot_table(index=self.period_index_col, columns=self.shopno_col, 
                                    values=self.outcome_col, aggfunc='mean'))
            
            # Если данные отсутствуют, возвращаем 0
            if y_control.empty:
                return 0.0
                
            # Вычисляем синтетический контроль
            synthetic_control = pd.Series(
                intercept + y_control.dot(unit_weights).values,
                index=y_control.index
            )
            
            # Находим RMSPE - оптимизируем вычисление
            common_periods = y_treat_mean.index.intersection(synthetic_control.index)
            if len(common_periods) == 0:
                return 0.0
            
            # Вычисляем ошибки прогноза быстрее через векторизованные операции
            errors = y_treat_mean.loc[common_periods] - synthetic_control.loc[common_periods]
            mse = np.mean(errors ** 2)
            return np.sqrt(mse)
        except Exception as e:
            print(f"Ошибка в rmspe: {str(e)}")
            return 0.0
    
    def plot_synthetic_diff_in_diff(self, T0):
        try:
            # Используем кэшированные значения, если они есть
            if self._att is None:
                att, unit_weights, time_weights, sdid_model_fit, intercept = self.synthetic_diff_in_diff()
            else:
                att = self._att
                unit_weights = self._unit_weights
                time_weights = self._time_weights
                sdid_model_fit = self._model
                intercept = self._intercept
            
            if sdid_model_fit is None:
                print("Не удалось построить синтетический diff-in-diff график из-за отсутствия модели")
                return
            
            # Оптимизация: используем pivot_table вместо pivot для повышения эффективности
            y_co_all = self.data.loc[self.data[self.treat_col] == 0] \
                        .pivot_table(index=self.period_index_col, columns=self.shopno_col,
                                    values=self.outcome_col, aggfunc='mean') \
                        .sort_index()
            sc_did = intercept + y_co_all.dot(unit_weights)
            
            treated_all = self.data.loc[self.data[self.treat_col] == 1] \
                            .groupby(self.period_index_col)[self.outcome_col].mean()
            
            pre_times = self.data.loc[self.data[self.period_index_col] < T0, self.period_index_col]
            post_times = self.data.loc[self.data[self.period_index_col] >= T0, self.period_index_col]
            avg_pre_period = pre_times.mean() if len(pre_times) > 0 else T0
            avg_post_period = post_times.mean() if len(post_times) > 0 else T0 + 1
            
            params = sdid_model_fit.params
            pre_sc = params.get("Intercept", 0)
            post_sc = pre_sc + params.get("after_treatment", 0)
            pre_treat = pre_sc + params.get("treated", 0)
            post_treat = post_sc + params.get("treated", 0) + params.get("after_treatment:treated", 0)
            
            sc_did_y0 = pre_treat + (post_sc - pre_sc)
            
            plt.style.use("ggplot")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})
            
            controls_all = self.data.loc[self.data[self.treat_col] == 0]
            for unit in controls_all[self.shopno_col].unique():
                subset = controls_all.loc[controls_all[self.shopno_col] == unit].sort_values(self.period_index_col)
                ax1.plot(subset[self.period_index_col], subset[self.outcome_col],
                        color="gray", alpha=0.5, linewidth=1)

            ax1.plot(sc_did.index, sc_did.values, label="Synthetic DID", color="black", alpha=0.8)
            ax1.plot(treated_all.index, treated_all.values, label="Тестовая группа", color="red", linewidth=2)
            
            ax1.plot([avg_pre_period, avg_post_period], [pre_sc, post_sc],
                    color="C5", label="Синтетический тренд", linewidth=2)
            ax1.plot([avg_pre_period, avg_post_period], [pre_treat, post_treat],
                    color="C2", label="Воздействие", linewidth=2)
            ax1.plot([avg_pre_period, avg_post_period], [pre_treat, sc_did_y0],
                    color="C2", linestyle="dashed", linewidth=2)
            
            x_bracket = avg_post_period
            y_top = post_treat
            y_bottom = sc_did_y0
            ax1.annotate(
                '', 
                xy=(x_bracket, y_bottom), 
                xytext=(x_bracket, y_top),
                arrowprops=dict(arrowstyle='|-|', color='purple', lw=2)
            )
            ax1.text(x_bracket + 0.5, (y_top + y_bottom) / 2, f"ATT = {round(att, 2)}",
                    color='purple', fontsize=12, va='center')
            
            ax1.legend()
            ax1.set_title("Синтетический diff-in-diff")
            ax1.axvline(T0, color='black', linestyle=':')
            ax1.set_ylabel(f"Значение {self.outcome_col}")

            ax2.bar(time_weights.index, time_weights.values, color='blue', alpha=0.7)
            ax2.axvline(T0, color="black", linestyle="dotted")
            ax2.set_ylabel("Веса для времени")
            ax2.set_xlabel("Время")
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Ошибка при построении графика: {str(e)}")
