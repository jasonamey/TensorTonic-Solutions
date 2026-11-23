import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a float.
    """
    top = float(math.sumprod(a,b))
    
    a_norm = float(math.hypot(*a))
    b_norm = float(math.hypot(*b))
    bottom = a_norm * b_norm
    
    if (a_norm == 0.0 or b_norm == 0.0): 
        return 0.0 
    return float(top/bottom)