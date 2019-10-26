# -*- coding: utf-8 -*-

from implementations import *
from proj1_helpers import *
from sigmoids import *


# compute cross validation for training, return the optimal weigths and theirs respective loss for the train and the
# test datas
def cross_validation_step_logistic_regression(y, tx, k_indices, k, gamma, degree, max_iter):

    # get k'th subgroup for the validation set, the others will be in the training set
    tx_train, tx_validation, y_train, y_validation = cross_validation_get_indices(y, tx, k_indices, k)

    # features expansion
    tx_train = build_poly(tx_train, degree)
    tx_validation = build_poly(tx_validation, degree)
    initial_w = np.zeros(tx_train.shape[1])

    # optimization with one of the methods in "implementations.py"
    weights, loss_train = logistic_regression(y_train, tx_train, initial_w, max_iter, gamma)

    # compute the loss for the train and test datas with the weigths found
    # ###loss_test = compute_mse(y_validation, tx_validation, weights)
    # ###loss_test = sigmoid_loss(y_validation, tx_validation, weights, logistic_function)

    # when optimizing by maximizing the accuracies, no need to compute the loss
    loss_test = np.NaN

    return weights, loss_train, loss_test


# compute the best hyperparameters for regularized optimization
def best_hyperparameters_logistic_regression(y, tx, degrees, gammas, k_fold, initial_w, max_iter, seed=1):

    # for each degree, store the best gamma and the respective loss
    losses = []
    gammas_best = []

    # build k indices for k-fold
    k_indices = build_k_indices(y, k_fold, seed)

    # compute cross validation with all lambdas for each degree
    for degree in degrees:

        # store the loss, respective to the lambdas
        losses_test = []

        # compute cross validation for each lambda of the specific degree
        for gamma in gammas:

            # to compute the total loss of each lambda by storing the loss for each iteration 
            # of the k-fold and computing the mean
            losses_test_tmp = []

            # perform cross-validation
            for k in range(k_fold):
                _, _, loss_test = cross_validation_step_logistic_regression(y, tx, k_indices, k, initial_w, gamma, degree, max_iter)
                losses_test_tmp.append(loss_test)

            # compute the loss for the specific lambda by taking the mean of the losses of each iteration of the k-fold
            losses_test.append(np.mean(losses_test_tmp))

        # find the optimal lambda hyperparameter by getting the minimum loss for each degree
        best_gamma_index = np.argmin(losses_test)
        gammas_best.append(gammas[best_gamma_index])
        losses.append(losses_test[best_gamma_index])

    # find the optimal degree hyperparameter by getting the minimum loss
    best_degree_index = np.argmin(losses)

    # compute the optimal hyperparameters
    opt_degree = degrees[best_degree_index]
    opt_gamma = gammas_best[best_degree_index]

    return opt_degree, opt_gamma


# compute the best hyperparameters for regularized optimization
def best_hyperparameters_accuracy_logistic_regression(y, tx, degrees, gamma, k_fold, max_iter, seed=1):

    # for each degree, store the best gamma and the respective accuracy
    gammas_best = []
    accuracies = []

    # build k indices for k-fold
    k_indices = build_k_indices(y, k_fold, seed)

    # compute cross validation with all lambdas for each degree
    for degree in degrees:

        # store the loss, respective to the lambdas
        accuracies_tmp = []

        # compute loss for each iteration of the k_fold
        for k in range(k_fold):
            weights, _, _ = cross_validation_step_logistic_regression(y, tx, k_indices, k, gamma, degree,
                                                                      max_iter)
            y_pred = predict_labels(weights, build_poly(tx, degree))
            accuracies_tmp.append(performance_measure(y_pred, y))

        accuracies.append(np.mean(accuracies_tmp))

    # find the optimal degree hyperparameter by getting the minimum loss
    best_degree_index = np.argmax(accuracies)

    # compute the optimal hyperparameters
    opt_degree = degrees[best_degree_index]

    return opt_degree


# compute the best hyperparameters for regularized optimization for each subset of the training dataset
#  and return the best weights of each subset, respective to the best hyperparameters
def train_models_logistic_regression(y, tx, degrees, gamma, k_fold,  max_iters, seed=1):
    # get the indices of each training subset
    indices_group = group_indices(tx)

    # store the best weights, degree and lambda for each training subset
    best_weights = []
    best_degree = []

    # compute the optimal hyperparameters for each training subset and the respective weights
    for indice_group in indices_group:
        y_subset = y[indice_group]
        tx_subset = drop_na_columns(tx[indice_group])

        # Standardize the training subset
        tx_subset_standardized, _, _ = standardize(tx_subset)

        opt_degree = best_hyperparameters_accuracy_logistic_regression(y_subset, tx_subset_standardized, degrees, gamma, k_fold, max_iters, seed)
        tx_subset_standardized_expanded = build_poly(tx_subset_standardized, opt_degree)
        weights, _ = logistic_regression(y_subset, tx_subset_standardized_expanded, np.zeros(tx_subset_standardized_expanded.shape[1]), max_iters, gamma)

        best_degree.append(opt_degree)
        best_weights.append(weights)

    return best_weights, best_degree
