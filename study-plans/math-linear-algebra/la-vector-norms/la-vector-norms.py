import numpy as np

def vector_norms(v: list) -> np.ndarray:
    """
    Returns a float64 array containing the L1, L2, and infinity norms.
    """
    v_arr = np.asarray(v)
    
    L1 = np.sum(np.abs(v_arr))
    L2 = np.linalg.norm(v_arr)
    L3 = np.max(np.abs(v_arr))

    return np.array((L1,L2,L3))
    