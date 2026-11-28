import numpy as np

def matrix_multiply(A: list, B: list) -> np.ndarray:
    """
    Returns the matrix product as a float64 array.
    """
    return np.array(A, dtype=np.float64) @ np.array(B, dtype=np.float64)
  