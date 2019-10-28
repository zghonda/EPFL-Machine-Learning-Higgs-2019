# -*- coding: utf-8 -*-

import numpy as np


def compute_mse(y, tx, w):
    """Function used to compute the loss."""
    # Compute the error
    e = y - tx @ w

    # Return the MSE
    return e.T @ e / (2 * len(y))


def sigmoid_loss(y, tx, w, sigmoid):
    """compute the loss by negative log likelihood with specific sigmoid function"""
    pred = sigmoid(tx@w)
    loss = y.T @ (np.log(pred)) + (1 - y).T @ (np.log(1 - pred))
    return - loss.item(0)
