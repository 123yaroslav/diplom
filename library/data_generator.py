import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def generate_gaussian_process_data(N=10, T=100, length_scale=2.0, seed=42):
    np.random.seed(seed)
    
    time_grid = np.linspace(0, 10, T)
    
    all_series = np.zeros((N, T))
    
    kernel = RBF(length_scale=length_scale)
    
    for i in range(N):
        gp = GaussianProcessRegressor(kernel=kernel, random_state=seed+i)
        X_train = time_grid.reshape(-1, 1)
        y_train = np.random.normal(0, 1, size=T)
        gp.fit(X_train, y_train)

        all_series[i, :] = gp.predict(time_grid.reshape(-1, 1))
    
    return all_series