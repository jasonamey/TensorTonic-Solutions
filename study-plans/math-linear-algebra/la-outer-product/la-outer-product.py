import numpy as np

def outer_product(u: list, v: list) -> np.ndarray:
    """
    Returns the float64 outer-product matrix.
    """
    
    return np.multiply.outer(u, v).astype(np.float64, copy=False)
