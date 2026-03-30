import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x_np = np.asarray(x)
    q_np = np.asarray(q)

    return np.percentile(x_np, q_np)