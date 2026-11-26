import numpy as np

def convexity_certificate(H: list) -> dict:
    """
    Returns the PSD decision and smallest eigenvalue in a dictionary.
    """
    e_vals = np.linalg.eigvalsh(H)

    return {"is_convex": np.all(e_vals >= 0) , "min_eigenvalue": round(float(e_vals[0]),6)}