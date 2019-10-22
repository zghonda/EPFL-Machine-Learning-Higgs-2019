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

    loss = np.sum(np.log(1 + np.exp(tx.dot(w)))) - y.T.dot(tx.dot(w))
    
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""

    #return gradient of log-likelihood loss
    gradient = tx.T.dot(sigmoid(tx.dot(w))-y)
    
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
    losses = []
    ws = []
       
    # build tx with adding constant 1 as first column
    x = np.c_[np.ones((y.shape[0], 1)), tx]
    w = np.insert(initial_w, 0, 1)
    #print(w.shape)
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        #print(w.shape)
        w, loss = learning_by_gradient_descent(y, x, w, gamma)
        #print(w.shape)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        ws.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

        
    return  w, loss
    


