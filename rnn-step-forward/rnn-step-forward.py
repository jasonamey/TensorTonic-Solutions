import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    return np.tanh(x_t @ Wx + h_prev @ Wh + b)



# import numpy as np

# def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
#     """
#     Performs a single forward step of a Vanilla RNN cell.
    
#     Inputs:
#     - x_t: Input data of shape (D,)
#     - h_prev: Previous hidden state of shape (H,)
#     - Wx: Weights for input-to-hidden of shape (D, H)
#     - Wh: Weights for hidden-to-hidden of shape (H, H)
#     - b: Biases of shape (H,)

#     Returns:
#     - h_t: Next hidden state of shape (H,)
#     - cache: Values needed for the backward pass
#     """
#     # 1. Dimension Validation (Safety first!)
#     assert x_t.shape[0] == Wx.shape[0], "Input dimension D mismatch"
#     assert h_prev.shape[0] == Wh.shape[0], "Hidden dimension H mismatch"

#     # 2. The Recurrent Calculation
#     # We use the Tanh activation to squash the sum into [-1, 1]
#     pre_activation = x_t @ Wx + h_prev @ Wh + b
#     h_t = np.tanh(pre_activation)
    
#     # 3. Cache for BPTT
#     # To learn, the network needs to remember what it saw during the forward pass
#     cache = (x_t, h_prev, Wx, Wh, h_t)

#     return h_t, cache