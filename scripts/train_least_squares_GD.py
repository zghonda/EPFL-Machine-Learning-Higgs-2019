# -*- coding: utf-8 -*-

from implementations import *
from proj1_helpers import *
from sigmoids import *


# compute cross validation for training, return the optimal weigths and theirs respective loss for the train and the
# test datas
def cross_validation_step_least_squares_GD(y, tx, k_indices, k, gamma, degree, max_iters):

    # get k'th subgroup for the validation set, the others will be in the training set
    tx_train, tx_validation, y_train, y_validation = cross_validation_get_indices(y, tx, k_indices, k)

    # features expansion
    tx_train = build_poly(tx_train, degree)
    tx_validation = build_poly(tx_validation, degree)

    # optimization with one of the methods in "implementations.py"
    weights, loss_train = least_squares_GD(y_train, tx_train, np.zeros(tx_train.shape[1]), max_iters, gamma)

    # compute the loss for the train and test datas with the weigths found
    # ###loss_test = compute_mse(y_validation, tx_validation, weights)

    # when optimizing by maximizing the accuracies, no need to compute the loss
    loss_test = np.NaN

    return weights, loss_train, loss_test


# compute the best hyperparameters for regularized optimization
def best_hyperparameters_least_squares_GD(y, tx, degrees, gamma, max_iters, k_fold, seed=1):

    # for each degree, store the best gamma and the respective loss
    losses = []

    # build k indices for k-fold
    k_indices = build_k_indices(y, k_fold, seed)

    # compute cross validation for each degree
    for degree in degrees:

        # store the loss, respective to the degree
        losses_tmp = []

        # perform cross-validation
        for k in range(k_fold):
            _, _, loss_test = cross_validation_step_least_squares_GD(y, tx, k_indices, k, gamma, degree, max_iters)
            losses_tmp.append(loss_test)

        losses.append(np.mean(losses_tmp))

    # find the optimal degree hyperparameter by getting the minimum loss
    best_degree_index = np.argmin(losses)

    # compute the optimal hyperparameters
    opt_degree = degrees[best_degree_index]

    return opt_degree


# compute the best hyperparameters for regularized optimization
def best_hyperparameters_accuracy_least_squares_GD(y, tx, degrees, gamma, max_iters, k_fold, seed=1):

    # for each degree, store the best respective accuracy
    accuracies = []

    # build k indices for k-fold
    k_indices = build_k_indices(y, k_fold, seed)

    # compute cross validation for each degree
    for degree in degrees:

        # store the accuracy, respective to the degree
        accuracies_tmp = []

        # compute loss for each iteration of the k_fold
        for k in range(k_fold):
            weights, _, _ = cross_validation_step_least_squares_GD(y, tx, k_indices, k, gamma, degree, max_iters)
            y_pred = predict_labels(weights, build_poly(tx, degree), logistic=False)
            accuracies_tmp.append(performance_measure(y_pred, y))

        accuracies.append(np.mean(accuracies_tmp))

    # find the optimal degree hyperparameter by getting the minimum loss
    best_degree_index = np.argmax(accuracies)

    # compute the optimal hyperparameters
    opt_degree = degrees[best_degree_index]

    return opt_degree


# compute the best hyperparameters for regularized optimization for each subset of the training dataset
#  and return the best weights of each subset, respective to the best hyperparameters
def train_models_least_squares_GD(y, tx, degrees, gamma, max_iters, k_fold, seed=1):
    # get the indices of each training subset
    indices_group = group_indices(tx)

    # store the best weights and degree for each training subset
    best_weights = []
    best_degree = []

    # compute the optimal hyperparameters for each training subset and the respective weights
    for indice_group in indices_group:
        y_subset = y[indice_group]
        tx_subset = drop_na_columns(tx[indice_group])

        # Standardize the training subset
        tx_subset_standardized, _, _ = standardize(tx_subset)

        opt_degree = best_hyperparameters_accuracy_least_squares_GD(y_subset, tx_subset_standardized, degrees, gamma, max_iters, k_fold, seed)
        tx_subset_standardized_expanded = build_poly(tx_subset_standardized, opt_degree)
        weights, _ = least_squares_GD(y_subset, tx_subset_standardized_expanded, np.zeros(tx_subset_standardized_expanded.shape[1]), max_iters, gamma)

        best_degree.append(opt_degree)
        best_weights.append(weights)

    return best_weights, best_degree
