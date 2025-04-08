import numpy as np

def generate_gaussian_process_data(N=10, T=30, length_scale=2.0, seed=42):
    """
    Генерируем (N) реализаций гауссовского процесса длины T
    (каждая – вектор длины T). Возвращаем массив shape (N, T).
    """
    np.random.seed(seed)
    t = np.arange(T)

    K = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            dist = (t[i] - t[j]) ** 2
            K[i, j] = np.exp(-dist / (2 * length_scale**2))

    data = np.random.multivariate_normal(mean=np.zeros(T), cov=K, size=N)
    return data