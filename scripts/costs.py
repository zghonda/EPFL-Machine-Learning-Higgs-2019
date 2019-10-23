# -*- coding: utf-8 -*-

def compute_mse(y, tx, w):
    """Function used to compute the loss."""
    # Compute the error
    e = y - tx @ w

    # Return the MSE
    return e.T @ e / (2 * len(y))
