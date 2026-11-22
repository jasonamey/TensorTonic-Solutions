import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    # x_arr = np.asarray(x)
    # y_arr = np.asarray(y)
    # return float(np.sum(x_arr * y_arr))

    return float(math.sumprod(x,y))
