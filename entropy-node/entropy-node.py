import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)

    if y.size == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    
    probabilities = counts / y.size

    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-15))

    return float(max(0.0, entropy))

