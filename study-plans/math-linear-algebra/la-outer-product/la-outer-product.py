import numpy as np

def outer_product(u: list, v: list) -> np.ndarray:
    """
    Returns the float64 outer-product matrix.
    """
    # u_arr = np.asarray(u, dtype=np.float64)
    # v_arr = np.asarray(v, dtype=np.float64)

    # return np.outer(u_arr, v_arr)
    return np.multiply.outer(u, v).astype(np.float64, copy=False)
