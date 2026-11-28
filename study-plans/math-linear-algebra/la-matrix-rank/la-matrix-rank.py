import numpy as np

def matrix_rank(A: list) -> int:
    """
    Returns the numerical rank as a Python integer.
    """
    a_arr = np.array(A, dtype=np.float64)
    singular_vals = np.linalg.svd(a_arr, compute_uv=False)
    e = np.finfo(np.float64).eps
    tau = np.max(a_arr.shape) * np.max(singular_vals) * e 
    results = np.where(singular_vals > tau, 1, 0)
    return np.sum(results)