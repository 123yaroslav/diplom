import numpy as np
import pandas as pd
import cvxpy as cp
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from scipy.stats import norm
import matplotlib.pyplot as plt


class SyntheticDIDModel:
    def __init__(self, data, metric, period_index, shopno, treated, after_treatment,
                 seed=42, bootstrap_rounds=100, njobs=4):
        self.data = data.copy()
        self.outcome_col = metric
        self.period_index_col = period_index
        self.shopno_col = shopno
        self.treat_col = treated
        self.post_col = after_treatment
        self.seed = seed
        self.bootstrap_rounds = bootstrap_rounds
        self.njobs = njobs

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
        joined = (data
                  .set_index([self.period_index_col, self.shopno_col])
                  .join(time_w)
                  .join(unit_w)
                  .reset_index()
                  .fillna({
                      time_w.name: 1 / len(pd.unique(data.loc[data[self.post_col] == 1, self.period_index_col])),
                      unit_w.name: 1 / len(pd.unique(data.loc[data[self.treat_col] == 1, self.shopno_col]))
                  })
                  .assign(**{"weights": lambda d: (d[time_w.name] * d[unit_w.name]).round(10)})
                  .astype({self.treat_col: int, self.post_col: int}))
        return joined

    def fit_time_weights(self, data):
        control = data.loc[data[self.treat_col] == 0]
        y_pre = (control
                 .loc[control[self.post_col] == 0]
                 .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col))
        y_post_mean = (control
                       .loc[control[self.post_col] == 1]
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
        zeta = self.calculate_regularization(data)
        pre_data = data.loc[data[self.post_col] == 0]
        y_pre_control = (pre_data
                         .loc[pre_data[self.treat_col] == 0]
                         .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col))
        y_pre_treat_mean = (pre_data
                            .loc[pre_data[self.treat_col] == 1]
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
        return pd.Series(w.value[1:], name="unit_weights", index=y_pre_control.columns), w.value[0]

    def synthetic_diff_in_diff(self, data=None):
        if data is None:
            data = self.data
        unit_weights, intercept = self.fit_unit_weights(data)
        time_weights = self.fit_time_weights(data)
        did_data = self.join_weights(data, unit_weights, time_weights)
        formula = f"{self.outcome_col} ~ {self.post_col}*{self.treat_col}"
        did_model = smf.wls(formula, data=did_data, weights=did_data["weights"] + 1e-10).fit()
        att = did_model.params[f"{self.post_col}:{self.treat_col}"]
        return att, unit_weights, time_weights, did_model, intercept

    def make_random_placebo(self, data):
        control = data.query(f"~{self.treat_col}")
        shopnos = control[self.shopno_col].unique()
        placebo_shopno = np.random.choice(shopnos)
        return control.assign(**{self.treat_col: control[self.shopno_col] == placebo_shopno})


    def _single_placebo_att(self):
        try:
            placebo_data = self.make_random_placebo(self.data)
            att_placebo, *_ = self.synthetic_diff_in_diff(data=placebo_data)
            return att_placebo
        except Exception as e:
            print(f"Ошибка в _single_placebo_att: {str(e)}")
            return np.nan

    def estimate_se(self, alpha=0.05):
        np.random.seed(self.seed)
        main_att, *_ = self.synthetic_diff_in_diff()

        effects = Parallel(n_jobs=self.njobs)(
            delayed(self._single_placebo_att)() for _ in range(self.bootstrap_rounds)
        )
        
        effects = [e for e in effects if not np.isnan(e)]
        
        if not effects:
            raise ValueError("Все бутстрэп-итерации завершились с ошибками")
            
        se = np.std(effects, ddof=1)
        z = norm.ppf(1 - alpha / 2)
        ci_lower = main_att - z * se
        ci_upper = main_att + z * se
        return main_att, se, ci_lower, ci_upper 
    
    def plot_synthetic_diff_in_diff(self, T0):
        att, unit_weights, time_weights, sdid_model_fit, intercept = self.synthetic_diff_in_diff()
        
        y_co_all = self.data.loc[self.data[self.treat_col] == 0] \
                      .pivot(index=self.period_index_col, columns=self.shopno_col, values=self.outcome_col) \
                      .sort_index()
        sc_did = intercept + y_co_all.dot(unit_weights)
        
        treated_all = self.data.loc[self.data[self.treat_col] == 1] \
                          .groupby(self.period_index_col)[self.outcome_col].mean()
        
        pre_times = self.data.loc[self.data[self.period_index_col] < T0, self.period_index_col]
        post_times = self.data.loc[self.data[self.period_index_col] >= T0, self.period_index_col]
        avg_pre_period = pre_times.mean() if len(pre_times) > 0 else T0
        avg_post_period = post_times.mean() if len(post_times) > 0 else T0 + 1
        
        params = sdid_model_fit.params
        pre_sc = params["Intercept"] if "Intercept" in params else 0
        post_sc = pre_sc + params["after_treatment"] if "after_treatment" in params else pre_sc
        pre_treat = pre_sc + params["treated"] if "treated" in params else pre_sc
        if "after_treatment:treated" in params:
            post_treat = post_sc + params["treated"] + params["after_treatment:treated"]
        else:
            post_treat = pre_treat
        
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
                 color="C5", label="Контрфактический тренд", linewidth=2)
        ax1.plot([avg_pre_period, avg_post_period], [pre_treat, post_treat],
                 color="C2", linestyle="dashed", label="Эффект", linewidth=2)
        ax1.plot([avg_pre_period, avg_post_period], [pre_treat, sc_did_y0],
                 color="C2", label="Синтетический тренд", linewidth=2)
        
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
