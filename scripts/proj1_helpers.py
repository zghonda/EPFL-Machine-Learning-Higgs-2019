# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def standardize_matrix(tx):
    return (tx - np.mean(tx)) / np.std(tx)


def build_poly(x, degree):
    """Performs polynomial extension."""
    polynom = np.ones((len(x), 1))
    for d in range(1, degree + 1):
        polynom = np.c_[polynom, np.power(x, d)]

    return polynom


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


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