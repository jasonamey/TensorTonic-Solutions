import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x_arr = np.array(x, dtype=float)
    p_arr = np.array(p, dtype=float) 

    if x_arr.shape != p_arr.shape:
        raise ValueError("Input x and probabilities p must have the same shape.")

    if not np.isclose(np.sum(p_arr), 1.0):
        raise ValueError("Probabilities must sum to 1.0")

    return np.dot(x_arr, p_arr)
    
    
