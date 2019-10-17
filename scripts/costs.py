# -*- coding: utf-8 -*-
import numpy as np

"""Function used to compute the loss."""


def compute_mse(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return (np.linalg.norm(e) ** 2) / len(y)
