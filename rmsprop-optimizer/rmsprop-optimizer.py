import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w, g, s = np.asarray(w), np.asarray(g), np.asarray(s)
    
    s_t = beta * s + (1 - beta) * np.square(g)

    return (w - (lr * g) / np.sqrt(s_t + eps), s_t) 
    

    