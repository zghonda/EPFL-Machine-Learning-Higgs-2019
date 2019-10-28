# -*- coding: utf-8 -*-

from costs import *
from gradient_descent import *
from helpers import *
from sigmoids import *


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


def least_squares(y, tx):
    """calculate the least squares solution."""
    w_star = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    loss = compute_mse(y, tx, w_star)
    return w_star, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    lambda_bis = lambda_ / (2 * len(y))
    fst_term = np.linalg.inv(tx.T @ tx + lambda_bis * np.identity(tx.shape[1]))
    snd_term = tx.T @ y
    w_star = fst_term @ snd_term

    # loss = compute_mse(y, tx, w_star) + lambda_ * np.linalg.norm(w_star) ** 2
    loss = np.NaN

    return w_star, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression using gradient descent"""
    w = initial_w
    sigmoid = logistic_function

    for iter in range(max_iters):
        gradient = compute_gradient_sigmoid(y, tx, w, sigmoid)
        w -= gamma * gradient

    # loss = sigmoid_loss(y, tx, w, sigmoid)
    loss = np.NaN

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    sigmoid = logistic_function

    for _ in range(max_iters):
        gradient = compute_gradient_sigmoid(y, tx, w, sigmoid) + lambda_ * w
        w -= gamma * gradient

    # loss = sigmoid_loss(y, tx, w, sigmoid) + lambda_ * (w.T@w).item(0) / 2
    loss = np.NaN

    return w, loss
