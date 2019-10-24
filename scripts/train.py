# -*- coding: utf-8 -*-

from costs import *
from helpers import *
from implementations import *
from proj1_helpers import *


# compute cross validation for training, return the optimal weigths and theirs respective loss for the train and the
# test datas
def cross_validation(y, tx, k_indices, k, lambda_, degree):
    # get k'th subgroup in test, others in train
    training_indices = k_indices[~(np.arange(len(k_indices)) == k)].reshape(-1)
    test_indices = k_indices[k]

    tx_train = tx[training_indices]
    tx_test = tx[test_indices]
    y_train = y[training_indices]
    y_test = y[test_indices]

    # features expansion
    tx_train = build_poly(tx_train, degree)
    tx_test = build_poly(tx_test, degree)

    # optimization with ridge_regression
    weigths, loss_train = ridge_regression(y_train, tx_train, lambda_)

    # compute the loss for the train and test datas with the weigths found
    loss_test = compute_mse(y_test, tx_test, weigths)

    return weigths, loss_train, loss_test


# compute the best hyperparameters for regularized optimization
def best_hyperparameters(y, tx, degrees, lambdas, k_fold, seed=1):
    # for each degree, store the best lambda and the respective loss
    losses = []
    lambdas_best = []

    # build k indices for k-fold
    k_indices = build_k_indices(y, k_fold, seed)

    # compute cross validation with all lambdas for each degree
    for degree in degrees:

        # store the loss, respective to the lambdas
        losses_test = []

        # compute cross validation for each lambda of the specific degree
        for lambda_ in lambdas:

            # to compute the total loss of each lambda by storing the loss for each iteration 
            # of the k-fold and computing the mean
            losses_test_tmp = []

            # compute loss for each iteration of the k_fold
            for k in range(k_fold):
                weigths, loss_train, loss_test = cross_validation(y, tx, k_indices, k, lambda_, degree)
                losses_test_tmp.append(loss_test)

            # compute the loss for the specific lambda by taking the mean of the losses of each iteration of the k-fold
            losses_test.append(np.mean(losses_test_tmp))

        # find the optimal lambda hyperparameter by getting the minimum loss for each degree
        best_lambda_index = np.argmin(losses_test)
        lambdas_best.append(lambdas[best_lambda_index])
        losses.append(losses_test[best_lambda_index])

    # find the optimal degree hyperparameter by getting the minimum loss
    best_degree_index = np.argmin(losses)

    # compute the optimal hyperparameters
    opt_degree = degrees[best_degree_index]
    opt_lambda = lambdas_best[best_degree_index]

    return opt_degree, opt_lambda


# compute the best hyperparameters for regularized optimization for each subset of the training dataset
#  and return the best weights of each subset, respective to the best hyperparameters
def train_models(y, tx, degrees, lambdas, k_fold, seed=1):
    # get the indices of each training subset
    indices_group = group_indices(tx)

    # store the best weights and degree for each training subset
    best_weights = []
    best_degree = []

    # compute the optimal hyperparameters for each training subset and the respective weights
    for indice_group in indices_group:
        y_subset = y[indice_group]
        tx_subset = drop_na_columns(tx[indice_group])

        opt_degree, opt_lambda = best_hyperparameters_accuracy(y_subset, tx_subset, degrees, lambdas, k_fold, seed)
        weights = ridge_regression(y_subset, build_poly(tx_subset, opt_degree), opt_lambda)

        best_degree.append(opt_degree)
        best_weights.append(weights)

    return best_weights, best_degree


# compute the best hyperparameters for regularized optimization
def best_hyperparameters_accuracy(y, tx, degrees, lambdas, k_fold, seed=1):
    # for each degree, store the best lambda and the respective loss
    ###losses = []
    lambdas_best = []
    accuracies = []

    # build k indices for k-fold
    k_indices = build_k_indices(y, k_fold, seed)

    # compute cross validation with all lambdas for each degree
    for degree in degrees:

        # store the loss, respective to the lambdas
        ###losses_test = []
        accuracies_lambda = []

        # compute cross validation for each lambda of the specific degree
        for lambda_ in lambdas:

            # to compute the total loss of each lambda by storing the loss for each iteration 
            # of the k-fold and computing the mean
            ###losses_test_tmp = []
            accuracies_tmp = []

            # compute loss for each iteration of the k_fold
            for k in range(k_fold):
                weights, _, _ = cross_validation(y, tx, k_indices, k, lambda_, degree)
                ###losses_test_tmp.append(loss_test)
                y_pred = predict_labels(weights, build_poly(tx, degree))
                accuracies_tmp.append(performance_measure(y_pred, y))

            # compute the loss for the specific lambda by taking the mean of the losses of each iteration of the k-fold
            accuracies_lambda.append(np.mean(accuracies_tmp))

        # find the optimal lambda hyperparameter by getting the minimum loss for each degree
        best_lambda_index = np.argmax(accuracies_lambda)
        lambdas_best.append(lambdas[best_lambda_index])
        accuracies.append(accuracies_lambda[best_lambda_index])

    # find the optimal degree hyperparameter by getting the minimum loss
    best_degree_index = np.argmax(accuracies)

    # compute the optimal hyperparameters
    opt_degree = degrees[best_degree_index]
    opt_lambda = lambdas_best[best_degree_index]

    return opt_degree, opt_lambda