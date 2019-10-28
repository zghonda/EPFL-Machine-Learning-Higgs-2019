# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from train_ridge_regression import *


# Load the training and testing data into feature matrix, class labels, and event ids

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


# Data preparation for model training
tX, tX_test = replace_data(tX, replace=-999, by=np.NaN), replace_data(tX_test, replace= -999, by=np.NaN)


# Parameters initialisation and definition

# The degrees to choose from for the polynomial expansion
degrees = np.arange(2, 8)

# The hyperparameters to choose from for regularized regressions
lambdas = np.logspace(-4, 0, 20)
gamma = 0.01

# The number of splitting part for the cross validation
k_fold = 4

# initialize the maximum number of iterations for updating the weights when using gradient descent
max_iters = 1000

# Get the indices of the subsets in the training and the test datasets
indices_train_group = group_indices(tX)
indices_test_group = group_indices(tX_test)

# initialize the prediction array
y_pred = np.zeros(tX_test.shape[0])

from train_ridge_regression import *

best_weights, best_degree, best_lambda = train_models_ridge_regression(y, tX, degrees, lambdas, k_fold)

for i, indice_test_group in enumerate(indices_test_group):
    # for standardizing the test subset, we need the data of both train and test subsets
    tx_subset = tX[indices_train_group]
    tx_test_subset = tX_test[indice_test_group]

    # get the standardized test subset
    _, standardized_tx_test_subset = preprocess_data(tx_subset, tx_test_subset)

    # predict the labels
    y_pred_subset = predict_labels(best_weights[i][0], build_poly(standardized_tx_test_subset, best_degree[i]), logistic=False)
    y_pred[indice_test_group] = y_pred_subset

OUTPUT_PATH = '../data/sample-submission_reg_log.csv'
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
