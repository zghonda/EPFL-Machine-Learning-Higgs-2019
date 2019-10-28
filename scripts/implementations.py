# -*- coding: utf-8 -*-
"""Implementations"""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the MSE loss"""   
    
    N = len(y)
    e = y-tx.dot(w)
    L = e.dot(e)  
    
    return L/N  


def compute_gradient(y, tx, w):
    """Compute the gradient.""" 
    
    N=len(y)
    e = y-tx.dot(w)
    del_L = np.transpose(tx).dot(e)   
    
    return -del_L/N


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm. 
       gamma = step size
    """
    #For debugging
    #ws = [initial_w]
    #losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        #get gradient
        gradient = compute_gradient(y, tx, w)    
        
        #update w by gradient
        w = w - gamma*gradient
        
        #For debugging
        #ws.append(w)
        #losses.append(loss)

    loss = compute_loss(y, tx, w)
    
    return w, loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just one sample chosen randomly."""
    
    #choose random point
    rand_sample=np.random.randint(len(y))
    
    #calculate gradient at the chosen point
    e = y[rand_sample]-tx[rand_sample].dot(w)
    del_L=np.transpose(tx[rand_sample]).dot(e)
    
    return -del_L
    
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    #For debugging
    #ws = [initial_w]
    #losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        #get gradient from single point
        gradient = compute_stoch_gradient(y, tx, w)

        #update w by gradient        
        w = w - gamma*gradient
        
        #For debugging
        #ws.append(w)
        #losses.append(loss)
       
    loss = compute_loss(y, tx, w)
    
    return w, loss


def least_squares(y, tx):
    """calculate the least squares solution."""

    #calculate optimal weights and its loss
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    loss = compute_loss(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    #calculate w and loss using ridge regression formulation
    
    N = len(y)
    w_ridge = np.linalg.inv(tx.T.dot(tx)+2*N*lambda_*np.identity(len(tx.T.dot(tx)))).dot(tx.T).dot(y)
    e = y-tx.dot(w_ridge)
    MSE_ridge = (e.dot(e))/N + lambda_*w_ridge.dot(w_ridge)
    
    return w_ridge, MSE_ridge 


def sigmoid(t):
    """apply sigmoid function on t."""
    
    #1/(1+e^(-x)) formula implemented to avoid overflow
    sigma = 1/(1 + np.exp(-t))

    return sigma


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""

    #return log likelihood loss
    loss = np.sum(np.log(1 + np.exp(tx.dot(w)))) - y.T.dot(tx.dot(w))
    #loss /= tx.shape[0]

    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""

    #return gradient of log-likelihood loss
    gradient = tx.T.dot(sigmoid(tx.dot(w))-y)
    #gradient /= tx.shape[0]

    
    return gradient

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """  
    grad = calculate_gradient(y, tx, w)
    
    w = w - gamma*grad
    
    loss = calculate_loss(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    prev_loss = 0
       
    # build tx with adding constant 1 as first column
    x = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.insert(initial_w, 0, 1)

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.

        w, loss = learning_by_gradient_descent(y, x, w, gamma)

        # For debug: log info
        #if iter % 100 == 0:
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss)
            
        # converging criterion, end loop if values are close
        prev_loss = loss
        if prev_loss != 0 and np.abs(loss - prev_loss) < threshold:
            break

        
    return  w, loss


def learning_by_penalized_gradient_descent(y, tx, w, lambda_, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """ 
   
    grad = calculate_gradient(y, tx, w) + lambda_*w
    
    w = w - gamma*grad
    
    loss = calculate_loss(y, tx, w) + (lambda_/2)*w.dot(w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    prev_loss = 0
       
    # build tx with adding constant 1 as first column
    x = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.insert(initial_w, 0, 1)

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        w, loss = learning_by_penalized_gradient_descent(y, x, w, lambda_, gamma)

        # For debug: log info
        #if iter % 100 == 0:
            #print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        
        # converging criterion, end loop if values are close        
        prev_loss = loss
        if prev_loss != 0 and np.abs(loss - prev_loss) < threshold:
            break
        
    return  w, loss



def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

import matplotlib.pyplot as plt


def cross_validation_gamma(y, tx):
    seed = 1
    k_fold = 2
    gammas = np.logspace(-2, 0.5, 50)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    initial_w =np.zeros(tx.shape[1])    
    test_setx = tx[k_indices]
    test_sety = y[k_indices]
    k_indices_not = [np.isin(np.random.permutation(y.shape[0]), k_indices[i], invert=True) for i in range(k_fold)]
    train_setx = np.asarray([tx[k_indices_not[i]] for i in range(k_fold)])
    train_sety = np.asarray([y[k_indices_not[i]] for i in range(k_fold)])

    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    #iterate over all gammas
    for i in range(len(gammas)):

        loss_train = []
        loss_test = []
        
        #train the training sets and calculate loss for training and test sets
        for k in range(k_fold):
            w_train = reg_logistic_regression(train_sety[k], train_setx[k], 0.01, initial_w, 1000, gammas[i])[0]            
            loss_train.append(calculate_loss(train_sety[k], train_setx[k], w_train[1:]))
            loss_test.append(calculate_loss(test_sety[k], test_setx[k], w_train[1:]))
        
        #average the mean loss for all K folds
        rmse_tr.append(np.mean(loss_train))
        rmse_te.append(np.mean(loss_test))

        
    # Sımple Plot
    plt.semilogx(gammas, rmse_tr, color='b' ,label = 'train', marker="|")
    plt.semilogx(gammas, rmse_te, color='r', label = 'test', marker="|")
    plt.ylabel('loss')
    plt.xlabel('gamma')
    plt.legend()
    plt.show()

    return rmse_tr, rmse_te, gammas

def cross_validation_lambda(y, tx):
    seed = 1
    k_fold = 5
    gamma = 1
    lambdas = np.logspace(-5, 0, 20)
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    initial_w =np.zeros(tx.shape[1])    
    test_setx = tx[k_indices]
    test_sety = y[k_indices]
    k_indices_not = [np.isin(np.random.permutation(y.shape[0]), k_indices[i], invert=True) for i in range(k_fold)]
    train_setx = np.asarray([tx[k_indices_not[i]] for i in range(k_fold)])
    train_sety = np.asarray([y[k_indices_not[i]] for i in range(k_fold)])
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    #iterate over all lambdas
    for i in range(len(lambdas)):

        loss_train = []
        loss_test = []
        
        #split the set into K-folds, train w/ regularized regression for single set and calculate loss for training and test sets
        for k in range(k_fold):
            w_train = reg_logistic_regression(train_sety[k], train_setx[k], lambdas[i], initial_w, 10000, gamma)[0]
            loss_train.append(learning_by_penalized_gradient_descent(train_sety[k], train_setx[k], w_train[1:], lambdas[i], gamma)[1])
            loss_test.append(learning_by_penalized_gradient_descent(test_sety[k], test_setx[k], w_train[1:], lambdas[i], gamma)[1])
        
        #store average loss
        rmse_tr.append(np.mean(loss_train))
        rmse_te.append(np.mean(loss_test))

        
    # Simple Plot    
    plt.semilogx(lambdas, rmse_tr, color='b' ,label = 'train', marker="|")
    plt.semilogx(lambdas, rmse_te, color='r', label = 'test', marker="|")
    plt.ylabel('penalized loss')
    plt.xlabel('lambda')
    plt.legend()
    plt.show()

    return rmse_tr, rmse_te, lambdas

def build_poly(x, degree):
    """Performs polynomial extension.  (Olivier)"""
    polynom = np.ones((len(x), 1))
    for d in range(1, degree + 1):
        polynom = np.c_[polynom, np.power(x, d)]

    return polynom


def cross_validation_degree(y, tx):
    seed = 1
    k_fold = 5
    gamma = 1
    degrees = np.array(range(10))
    lambda_=0.01
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    initial_w =np.zeros(tx.shape[1])    
    test_setx = tx[k_indices]
    test_sety = y[k_indices]
    k_indices_not = [np.isin(np.random.permutation(y.shape[0]), k_indices[i], invert=True) for i in range(k_fold)]
    train_setx = np.asarray([tx[k_indices_not[i]] for i in range(k_fold)])
    train_sety = np.asarray([y[k_indices_not[i]] for i in range(k_fold)])
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []

    for i in degrees:

        loss_train = []
        loss_test = []

        for k in range(k_fold):
            
            w_train = reg_logistic_regression(train_sety[k], build_poly(train_setx[k], i), lambda_, np.zeros(1+i*tx.shape[1]), 1000, gamma)[0]
            loss_train.append(calculate_loss(train_sety[k], build_poly(train_setx[k],i), w_train[1:]))
            loss_test.append(calculate_loss(test_sety[k], build_poly(test_setx[k],i), w_train[1:]))
        
        
        rmse_tr.append(np.mean(loss_train))
        rmse_te.append(np.mean(loss_test))

        
    # ***************************************************    
    plt.scatter(degrees, rmse_tr, color='b' ,label = 'train', marker="|")
    plt.scatter(degrees, rmse_te, color='r', label = 'test', marker="|")
    plt.ylabel('penalized loss')
    plt.xlabel('degrees')
    plt.legend()
    plt.show()

    return rmse_tr, rmse_te, degrees

def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim([0, 1])
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")    
    

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    
    number_data = int(len(y)*ratio)
    chosen_data=np.random.choice(len(y), number_data, replace=False)
    split_x = x[chosen_data]
    split_y = y[chosen_data]
    
    return split_x, split_y   


def bias_variance_demo(y, tx):
    """The entry."""
    # define parameters
    seeds = range(1000)
    num_data = 10000
    ratio_train = 0.005
    degrees = range(5)
    
    # define list to store the variable
    rmse_tr = np.empty((len(seeds), len(degrees)))
    rmse_te = np.empty((len(seeds), len(degrees)))
    
    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)

        # split data with a specific seed
        
        split_x, split_y = split_data(tx, y, ratio_train, seed)
        
        w = [logistic_regression(split_y, build_poly(split_x, degrees[i]), np.ones(degrees[i]*split_x.shape[1]+1), 100, 0.9)[0] for i in range(len(degrees))]
        
        for j in range(len(degrees)):

            rmse_tr[index_seed, j] = np.sqrt(2*(calculate_loss(split_y, build_poly(split_x, degrees[j]),w[j][1:])))
            rmse_te[index_seed, j] = np.sqrt(2*(calculate_loss(y, build_poly(tx, degrees[j]),w[j][1:])))
        
    bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_tr)
    
    
    return rmse_tr, rmse_te



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


def drop_na_columns(x):
    """Drops all columns that have only NaN values"""
    result = x[:, ~np.all(np.isnan(x), axis=0)]
    return result