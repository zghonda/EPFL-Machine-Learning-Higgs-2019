# -*- coding: utf-8 -*-
import numpy as np


from costs import compute_mse
from gradient_descent import *
from proj1_helpers import predict_labels


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        for mini_batch_y, mini_batch_tx in batch_iter(y, tx, 1):
            gradient = compute_stoch_gradient(mini_batch_y, mini_batch_tx, w)
            w = w - gamma * gradient

    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
	"""implement ridge regression"""
	lambda_bis = 2  * lambda_ * tx.shape[0]
	a = tx.T@tx + lambda_bis * np.identity(tx.shape[1])
	b = tx.T@y
	return np.linalg.solve(a, b)