from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import cvxpy as cp

class SyntheticControl(BaseEstimator, RegressorMixin):
    def __init__(self, add_constant=True):
        self.add_constant = add_constant

    def fit(self, X, y):
        # Validate inputs
        X, y = check_X_y(X, y)
        n_features = X.shape[1]

        if self.add_constant:
            # Introduce variable a for the constant and w for the weights
            a = cp.Variable()
            w = cp.Variable(n_features)
            # The regression model now becomes: prediction = a + X @ w
            objective = cp.Minimize(cp.sum_squares(a + X @ w - y))
            # The weights still satisfy a convex combination constraint
            constraints = [cp.sum(w) == 1, w >= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False)
            self.a_ = a.value
            self.w_ = w.value
        else:
            # Fallback to the original model without the constant term.
            w = cp.Variable(n_features)
            objective = cp.Minimize(cp.sum_squares(X @ w - y))
            constraints = [cp.sum(w) == 1, w >= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False)
            self.w_ = w.value
            self.a_ = 0.0  # Set constant to zero if not used

        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # Ensure the estimator has been fitted before making predictions
        check_is_fitted(self)
        X = check_array(X)
        return self.a_ + X @ self.w_
