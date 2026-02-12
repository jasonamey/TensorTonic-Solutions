import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
 
    a = np.asarray(a)
    b = np.asarray(b)

    if a.shape != b.shape:
        raise ValueError(f"Input shapes {a.shape} and {b.shape} must match.")

    EPS = 1e-8

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < EPS or norm_b < EPS: 
        return 0.0

    dot_product = np.dot(a,b)

    similarity = dot_product / (norm_a * norm_b)

    return np.clip(similarity, -1.0, 1.0)