# -*- coding: utf-8 -*-
import numpy as np

"""Function used to compute the loss."""


#Compute the Mean Square Error
def compute_mse(y, tx, w):

	#Compute the error
    e = y - tx@w

    #Return the MSE
    return e.T@e / (2 * len(y))
