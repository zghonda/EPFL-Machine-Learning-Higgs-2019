# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """Performs polynomial extension."""
    polynom = np.ones((len(x), 1))
    for d in range(1, degree + 1):
        polynom = np.c_[polynom, np.power(x, d)]

    return polynom


def group_indices(x):
    """get the indices of the six groups.
    Each column contains the range of indices of each group
    There 3 main groups of particles divided according to their PRI_jet_num
    and each main group is divided to two subgroups depending on if they have a DER_mass_MMC or not."""

    # g0_ind_wo_mmc means indices of group 0 of the particles that don't have a DER_mass_MMC

    g0_ind_wo_mmc = np.where((x[:, 22] == 0) & (np.isnan(x[:, 0])))
    g0_ind_w_mmc = np.where((x[:, 22] == 0) & (~np.isnan(x[:, 0])))
    g1_ind_wo_mmc = np.where((x[:, 22] == 1) & (np.isnan(x[:, 0])))
    g1_ind_w_mmc = np.where((x[:, 22] == 1) & (~np.isnan(x[:, 0])))
    g2_ind_wo_mmc = np.where((x[:, 22] >= 2) & (np.isnan(x[:, 0])))
    g2_ind_w_mmc = np.where((x[:, 22] >= 2) & (~np.isnan(x[:, 0])))

    return [g0_ind_wo_mmc, g0_ind_w_mmc, g1_ind_wo_mmc, g1_ind_w_mmc, g2_ind_wo_mmc, g2_ind_w_mmc]


def replace_data(x, replace, by):
    return np.where(x == replace, by, x)


def drop_na_columns(x):
    """Drops all columns that have only NaN values"""
    result = x[:, ~np.all(np.isnan(x), axis=0)]
    return result


def preprocess_data(x_train, x_test):
    """ Apply data pre processing and cleaning """
    # data standardization
    x_train, mean_tr, std_tr = standardize(x_train)
    x_test = (x_test - mean_tr) / std_tr  # ref : https://bit.ly/2MytlBg

    # drop columns that have only Nan values
    x_train, x_test = drop_na_columns(x_train), drop_na_columns(x_test)

    return x_train, x_test


def performance_measure(y_pred, y):
    """ Measures the accuracy of the predictions"""
    diff = y_pred - y
    n_correct = diff[diff == 0].shape[0]
    return n_correct / len(y)
