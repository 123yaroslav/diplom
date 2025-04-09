from causalpy.pymc_models import PyMCModel
from causalpy.utils import round_num
import pymc as pm
import numpy as np
import arviz as az


class WeightedSumFitter(PyMCModel):
    def build_model(self, X, y, coords):
        with self:
            n_predictors = X.shape[1]
            X = pm.Data("X", X, dims=["obs_ind", "coeffs"])
            y = pm.Data("y", y[:, 0], dims="obs_ind")
            has_intercept = 'Intercept' in coords.get("coeffs", [])
            if has_intercept:
                intercept = pm.Normal("intercept", mu=0, sigma=1)
                beta = pm.Dirichlet("beta", a=np.ones(n_predictors-1), dims="coeffs")
                X_without_intercept = X[:, 1:] 
                mu = pm.Deterministic("mu", intercept + pm.math.dot(X_without_intercept, beta), dims="obs_ind")
            else:
                beta = pm.Dirichlet("beta", a=np.ones(n_predictors), dims="coeffs")
                mu = pm.Deterministic("mu", pm.math.dot(X, beta), dims="obs_ind")
            
            sigma = pm.HalfNormal("sigma", 1)
            pm.Normal("y_hat", mu, sigma, observed=y, dims="obs_ind")


    def print_coefficients(self, labels, round_to=None):
        """Print the model coefficients."""
        print("Model coefficients:")

        coeffs = az.extract(self.idata.posterior, var_names=["beta"])
        
        max_label_length = max(len(name) for name in labels)
        
        has_intercept = "intercept" in self.idata.posterior.variables
        
        for i, name in enumerate(labels):
            if has_intercept and name == "Intercept":
                coeff_samples = az.extract(self.idata.posterior, var_names=["intercept"])
            else:
                idx = i-1 if has_intercept else i
                coeff_samples = coeffs.isel(coeffs=idx)
            
            formatted_name = f"  {name: <{max_label_length}}"
            formatted_val = f"{round_num(coeff_samples.mean().data, round_to)}, 94% HDI [{round_num(coeff_samples.quantile(0.03).data, round_to)}, {round_num(coeff_samples.quantile(1 - 0.03).data, round_to)}]"
            print(f"  {formatted_name}  {formatted_val}")
        
        coeff_samples = az.extract(self.idata.posterior, var_names=["sigma"])
        name = "sigma"
        formatted_name = f"  {name: <{max_label_length}}"
        formatted_val = f"{round_num(coeff_samples.mean().data, round_to)}, 94% HDI [{round_num(coeff_samples.quantile(0.03).data, round_to)}, {round_num(coeff_samples.quantile(1 - 0.03).data, round_to)}]"
        print(f"  {formatted_name}  {formatted_val}")