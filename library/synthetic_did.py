import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from toolz import partial

class SyntheticDIDModel:
    def __init__(self, data, outcome_col, period_index_col, shopno_col, treat_col, post_col, seed=0):
        """
        Параметры:
         - data: исходный pandas DataFrame
         - outcome_col: название столбца с исходом (например, "mean_delivery")
         - period_index_col: название столбца с индексом периода (например, "period_index")
         - shopno_col: название столбца с идентификатором магазина (например, "shopno")
         - treat_col: название столбца, обозначающего лечение (например, "treated")
         - post_col: название столбца, обозначающего пост-период (например, "after_treatment")
         - bootstrap_rounds: число раундов бутстрепа для оценки стандартной ошибки
         - seed: зерно генератора случайных чисел
         - njobs: число параллельных задач
        """
        self.data = data.copy()
        self.outcome_col = outcome_col
        self.period_index_col = period_index_col
        self.shopno_col = shopno_col
        self.treat_col = treat_col
        self.post_col = post_col
        self.seed = seed

    def make_random_placebo(self, data):
        """
        Создаёт placebo эффект для бутстрепа
        """
        control = data.query(f"~{self.treat_col}")
        shopnos = control[self.shopno_col].unique()
        placebo_shopno = np.random.choice(shopnos)
        return control.assign(**{self.treat_col: control[self.shopno_col] == placebo_shopno})

    def calculate_regularization(self, data):
        """
        Вычисляет параметр регуляризации zeta
        """
        n_treated_post = data.query(f"{self.post_col}").query(f"{self.treat_col}").shape[0]

        first_diff_std = (data
                          .query(f"~{self.post_col}")
                          .query(f"~{self.treat_col}")
                          .sort_values(self.period_index_col)
                          .groupby(self.shopno_col)[self.outcome_col]
                          .diff()
                          .std())

        return n_treated_post ** (1 / 4) * first_diff_std

    def join_weights(self, data, unit_w, time_w):
        """
        Объединяет веса по времени и по единицам в исходный DataFrame для дальнейшего проведения WLS
        """
        joined = (data
                  .set_index([self.period_index_col, self.shopno_col])
                  .join(time_w)
                  .join(unit_w)
                  .reset_index()
                  .fillna({
                      time_w.name: 1 / len(pd.unique(data.query(f"{self.post_col}")[self.period_index_col])),
                      unit_w.name: 1 / len(pd.unique(data.query(f"{self.treat_col}")[self.shopno_col]))
                  })
                  .assign(**{"weights": lambda d: (d[time_w.name] * d[unit_w.name]).round(10)})
                  .astype({self.treat_col: int, self.post_col: int}))
        return joined

    def fit_time_weights(self, data):
        """
        Оценка весов для временных периодов
        """
        control = data.query(f"~{self.treat_col}")
        y_pre = (control
                 .query(f"~{self.post_col}")
                 .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col))
        y_post_mean = (control
                       .query(f"{self.post_col}")
                       .groupby(self.shopno_col)[self.outcome_col]
                       .mean()
                       .values)
        X = np.concatenate([np.ones((1, y_pre.shape[1])), y_pre.values], axis=0)

        w = cp.Variable(X.shape[0])
        objective = cp.Minimize(cp.sum_squares(w @ X - y_post_mean))
        constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)

        return pd.Series(w.value[1:], name="time_weights", index=y_pre.index)

    def fit_unit_weights(self, data):
        """
        Оценка весов для контрольных единиц
        """
        zeta = self.calculate_regularization(data)
        pre_data = data.query(f"~{self.post_col}")

        y_pre_control = (pre_data
                         .query(f"~{self.treat_col}")
                         .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col))

        y_pre_treat_mean = (pre_data
                            .query(f"{self.treat_col}")
                            .groupby(self.period_index_col)[self.outcome_col]
                            .mean())

        T_pre = y_pre_control.shape[0]
        X = np.concatenate([np.ones((T_pre, 1)), y_pre_control.values], axis=1)

        w = cp.Variable(X.shape[1])
        objective = cp.Minimize(cp.sum_squares(X @ w - y_pre_treat_mean.values) +
                                T_pre * zeta ** 2 * cp.sum_squares(w[1:]))
        constraints = [cp.sum(w[1:]) == 1, w[1:] >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)

        return pd.Series(w.value[1:], name="unit_weights", index=y_pre_control.columns)

    def synthetic_diff_in_diff(self, data=None):
        """
        Основной метод для оценки эффекта с помощью синтетического difference-in-differences
        Если data не передан, используется self.data
        """
        if data is None:
            data = self.data

        unit_weights = self.fit_unit_weights(data)
        time_weights = self.fit_time_weights(data)

        did_data = self.join_weights(data, unit_weights, time_weights)

        formula = f"{self.outcome_col} ~ {self.post_col}*{self.treat_col}"
        did_model = smf.wls(formula, data=did_data, weights=did_data["weights"] + 1e-10).fit()

        return did_model.params[f"{self.post_col}:{self.treat_col}"], unit_weights, time_weights

    def estimate_se(self):
        """
        Оценка стандартной ошибки эффекта через бутстреп
        """
        np.random.seed(self.seed)

        effects = Parallel(n_jobs=self.njobs)(
            delayed(self.synthetic_diff_in_diff)(self.make_random_placebo(self.data))
            for _ in range(self.bootstrap_rounds)
        )
        return np.std(effects, axis=0)
