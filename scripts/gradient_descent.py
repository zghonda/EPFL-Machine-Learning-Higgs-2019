# -*- coding: utf-8 -*-


def compute_gradient(y, tx, w):
    e = y - tx@w
    return - tx.T@e / (len(y))

def compute_stoch_gradient(y, tx, w):
    return compute_gradient(y, tx, w)