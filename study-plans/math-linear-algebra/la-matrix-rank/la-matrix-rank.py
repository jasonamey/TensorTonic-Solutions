import numpy as np

def matrix_rank(A: list) -> int:
    """
    Returns the numerical rank as a Python integer.
    """
    a_arr = np.array(A, dtype=np.float64)
    singular_vals = np.linalg.svd(a_arr, compute_uv=False)
    e = np.finfo(np.float64).eps
    tau = np.max(a_arr.shape) * singular_vals[0] * e 
    return int(np.sum(singular_vals > tau))