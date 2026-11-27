import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a Python float.
    """

    A_arr = np.asarray(A)
    total = 0.0 

    for i in np.arange(A_arr.shape[0]):
        total += A_arr[i,i]

    return total 
    