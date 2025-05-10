import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp


class SyntheticControl:
    def __init__(self, data, metric, period_index, shopno, treated, after_treatment, bootstrap_rounds=100, seed=42, intercept=False):
        self.data = data.copy()
        self.metric = metric
        self.period_index = period_index
        self.shopno = shopno
        self.treated = treated
        self.after_treatment = after_treatment
        self.bootstrap_rounds = bootstrap_rounds
        self.seed = seed
        self.intercept = intercept

    def make_random_placebo(self):
        control = self.data.query(f"~{self.treated}")
        shopnos = control[self.shopno].unique()
        placebo_shopno = np.random.choice(shopnos)
        return control.assign(**{self.treated: control[self.shopno] == placebo_shopno})

    def _loss_function(self, params, X, y):
        """Функция потерь для оптимизатора"""
        if self.intercept:
            a, w = params[0], params[1:]
            pred = a + X @ w
        else:
            w = params
            pred = X @ w
        
        return np.sum((pred - y) ** 2)

    def _weight_constraint(self, params):
        """Ограничение: сумма весов = 1"""
        if self.intercept:
            return np.sum(params[1:]) - 1
        else:
            return np.sum(params) - 1

    def synthetic_control(self, data=None):
        if data is None:
            data = self.data
        
        is_original_data = (data is self.data)  # Флаг, определяющий оригинальные данные

        x_pre_control = (data
                         .query(f"not {self.treated}")
                         .query(f"not {self.after_treatment}")
                         .pivot(index=self.period_index, columns=self.shopno, values=self.metric)
                         .values)

        y_pre_treat_mean = (data
                            .query(f"not {self.after_treatment}")
                            .query(f"{self.treated}")
                            .groupby(self.period_index)[self.metric]
                            .mean())

        n_controls = x_pre_control.shape[1]
        
        # Проверка на пустые данные
        if n_controls == 0 or len(y_pre_treat_mean) == 0:
            raise ValueError("Недостаточно данных для обучения синтетического контроля")
        
        if self.intercept:
            # Начальные значения: интерсепт = 0, равные веса
            params_init = np.zeros(n_controls + 1)
            params_init[1:] = 1.0 / n_controls
            bounds = [(None, None)] + [(0.0, None)] * n_controls
        else:
            # Начальные значения: равные веса
            params_init = np.ones(n_controls) / n_controls
            bounds = [(0.0, None)] * n_controls
            
        try:
            result = fmin_slsqp(
                lambda params: self._loss_function(params, x_pre_control, y_pre_treat_mean.values),
                params_init,
                f_eqcons=self._weight_constraint,
                bounds=bounds,
                disp=0
            )
            
            if self.intercept:
                a_value, w_value = result[0], result[1:]
            else:
                a_value, w_value = 0, result
                
            # Сохраняем вычисленные веса только для оригинальных данных
            if is_original_data:
                self.w_ = w_value
                if self.intercept:
                    self.a_ = a_value
            
            # Получаем контрольные данные для всех периодов
            x_all_control = (data
                            .query(f"not {self.treated}")
                            .pivot(index=self.period_index, columns=self.shopno, values=self.metric)
                            .values)

            if self.intercept:
                sc_series = a_value + x_all_control @ w_value
            else:
                sc_series = x_all_control @ w_value

            y_post_treat = data.query(f"{self.treated} and {self.after_treatment}")[self.metric].values
            sc_post = sc_series[-len(y_post_treat):]
            att = np.mean(y_post_treat - sc_post)
            return att
            
        except Exception as e:
            raise RuntimeError(f"Не удалось решить задачу оптимизации: {str(e)}")
    
    def rmspe(self):
        if not hasattr(self, 'w_'):
            # Если веса еще не вычислены, вызываем synthetic_control() с оригинальными данными
            self.synthetic_control(data=self.data)

        pre_data = self.data.query(f"not {self.after_treatment}")

        X_pre = (pre_data
                 .query(f"not {self.treated}")
                 .pivot(index=self.period_index, columns=self.shopno, values=self.metric)
                 .values)
        y_pre = (pre_data
                 .query(f"{self.treated}")
                 .sort_values(self.period_index)[self.metric]
                 .values)

        # Проверяем согласованность размерностей перед умножением
        if X_pre.shape[1] != self.w_.shape[0]:
            raise ValueError(f"Несоответствие размерностей матриц: X_pre {X_pre.shape}, w_ {self.w_.shape}")

        synth_pre = X_pre.dot(self.w_)
        if self.intercept and hasattr(self, 'a_'):
            synth_pre = self.a_ + synth_pre

        error = y_pre - synth_pre
        return np.sqrt(np.mean(error**2))
    
    def estimate_se_sc(self, alpha=0.05):
        np.random.seed(self.seed)
        
        # Вычисляем ATT для оригинальных данных
        att = self.synthetic_control(data=self.data)

        effects = []
        for _ in range(self.bootstrap_rounds):
            placebo_data = self.make_random_placebo()
            # Используем синтетический контроль для плацебо-данных без изменения атрибутов
            att_placebo = self.synthetic_control(data=placebo_data)
            effects.append(att_placebo)
        
        se = np.std(effects, ddof=1)

        z = norm.ppf(1 - alpha / 2)
        ci_lower = att - z * se
        ci_upper = att + z * se

        return se, ci_lower, ci_upper
    
    def plot_synthetic_control(self, T0):
        y_co_pre = (self.data
                    .query(f"{self.treated} == False and {self.after_treatment} == False")
                    .pivot(index=self.period_index, columns=self.shopno, values=self.metric)
                   )
        y_tr_pre = (self.data
                    .query(f"{self.treated} == True and {self.after_treatment} == False")
                    .sort_values(self.period_index)[self.metric]
                   )
        X = y_co_pre.values
        y = y_tr_pre.values
        n_features = X.shape[1]
        
        # Используем fmin_slsqp вместо cvxpy
        if self.intercept:
            params_init = np.zeros(n_features + 1)
            params_init[1:] = 1.0 / n_features
            bounds = [(None, None)] + [(0.0, None)] * n_features
        else:
            params_init = np.ones(n_features) / n_features
            bounds = [(0.0, None)] * n_features
            
        result = fmin_slsqp(
            lambda params: self._loss_function(params, X, y),
            params_init,
            f_eqcons=self._weight_constraint,
            bounds=bounds,
            disp=0
        )
        
        if self.intercept:
            self.a_ = result[0]
            self.w_ = result[1:]
        else:
            self.w_ = result
        
        self.control_units_ = y_co_pre.columns

        y_co_all = (self.data
                    .query(f"{self.treated} == False")
                    .pivot(index=self.period_index, columns=self.shopno, values=self.metric)
                    .sort_index()
                   )
        weights_series = pd.Series(self.w_, index=self.control_units_, name="weights")
        if self.intercept:
            sc_full = self.a_ + y_co_all.dot(weights_series)
        else:
            sc_full = y_co_all.dot(weights_series)

        y_tr_all = (self.data
                    .query(f"{self.treated} == True")
                    .sort_values(self.period_index)[self.metric]
                    .reset_index(drop=True)
                   )

        print("Веса синтетического контроля:")
        weights_df = weights_series.reset_index().rename(columns={'index': 'unit'})
        for i in range(len(weights_df)):
            value = weights_df.loc[i, 'weights']
            if value > 0:
                print(f"Индекс: {weights_df.loc[i, 'unit']}, Значение: {round(value, 2)}")

        fig, ax = plt.subplots(figsize=(14, 7))

        controls_all = self.data.query(f"{self.treated} == False")
        for unit_idx in controls_all[self.shopno].unique():
            subset = controls_all.query(f"{self.shopno} == @unit_idx").sort_values(self.period_index)
            ax.plot(subset[self.period_index], subset[self.metric], color="gray", alpha=0.5, linewidth=1)

        treated_all = self.data.query(f"{self.treated} == True").sort_values(self.period_index)
        ax.plot(treated_all[self.period_index], treated_all[self.metric], color="red", label="Тестовая группа", linewidth=2)

        ax.plot(sc_full.index, sc_full.values, color="black", linestyle="--",
                label="Синтетический контроль", linewidth=2)

        ax.axvline(T0 - 0.5, color='blue', linestyle=':', label='Начало вмешательства')
        
        ax.set_xlabel("Время")
        ax.set_ylabel(f"Значение {self.metric}")
        ax.set_title("Синтетический контроль")
        ax.legend()
        plt.tight_layout()
        plt.show()
