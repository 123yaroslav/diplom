import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


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


    def synthetic_control(self, data=None):
        if data is None:
            data = self.data

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

        w = cp.Variable(x_pre_control.shape[1])
        if self.intercept:
            a = cp.Variable()
            objective = cp.Minimize(cp.sum_squares(a + x_pre_control @ w - y_pre_treat_mean.values))
        else:
            objective = cp.Minimize(cp.sum_squares(x_pre_control @ w - y_pre_treat_mean.values))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        x_all_control = (data
                         .query(f"not {self.treated}")
                         .pivot(index=self.period_index, columns=self.shopno, values=self.metric)
                         .values)

        if self.intercept:
            sc_series = a.value + x_all_control @ w.value
        else:
            sc_series = x_all_control @ w.value

        y_post_treat = data.query(f"{self.treated} and {self.after_treatment}")[self.metric].values
        sc_post = sc_series[-len(y_post_treat):]
        att = np.mean(y_post_treat - sc_post)
        return att
    
    def estimate_se_sc(self, alpha=0.05):
        np.random.seed(self.seed)
        att = self.synthetic_control()

        effects = []
        for _ in range(self.bootstrap_rounds):
            placebo_data = self.make_random_placebo()
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
        
        w = cp.Variable(n_features)
        if self.intercept:
            a = cp.Variable()
            objective = cp.Minimize(cp.sum_squares(a + X @ w - y))
        else:
            objective = cp.Minimize(cp.sum_squares(X @ w - y))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        if self.intercept:
            self.a_ = a.value
        
        self.w_ = w.value
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

        fig, ax = plt.subplots(figsize=(10, 6))

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
