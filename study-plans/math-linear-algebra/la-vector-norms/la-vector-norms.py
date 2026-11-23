import numpy as np

def vector_norms(v: list) -> np.ndarray:
    """
    Returns a float64 array containing the L1, L2, and infinity norms.
    """
    v_arr = np.asarray(v, dtype=np.float64)
    v_abs = np.abs(v_arr)
    
    L1 = np.sum(v_abs)
    L2 = np.sqrt(v_arr @ v_arr)
    L3 = np.max(v_abs)

    out = np.empty(3,dtype=np.float64)

    out[0], out[1], out[2] = L1, L2, L3 

    return out
    