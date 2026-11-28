import numpy as np

def matrix_determinant(A: list) -> float:
    """
    Returns the determinant as a Python float.
    """
    a_arr = np.array(A, dtype=np.float64)
    return np.linalg.det(a_arr)