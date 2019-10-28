

import numpy as np


def logistic_function(t):
    return 1 / (1 + np.exp(-t))
