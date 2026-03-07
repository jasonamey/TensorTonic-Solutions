import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """

    A_copy = np.array(A, dtype=float)
    
    rows, cols = A_copy.shape

    m = np.empty((cols, rows), dtype=float)

    for i, row in enumerate(A):
        for j, value in enumerate(row):
            m[j,i] = value

    return m