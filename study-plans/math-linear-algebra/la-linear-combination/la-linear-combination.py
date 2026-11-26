import numpy as np

def linear_combination(vectors: list, coefficients: list) -> np.ndarray:
    """
    Returns the weighted sum as a float64 vector.
    """

    v_arr = np.asarray(vectors, dtype=np.float64)
    c_arr = np.asarray(coefficients, dtype=np.float64)

    return c_arr @ v_arr