# -*- coding: utf-8 -*-
"""Gradient Descent"""


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx @ w
    n = len(y)
    return (-1 / n) * tx.T @ e


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y, tx, w)
