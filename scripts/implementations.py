# -*- coding: utf-8 -*-
"""Implementations"""
import numpy as np



def compute_loss(y, tx, w):
    """Calculate the loss.
    MSE
    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    
    N = len(y)
    e = y-tx.dot(w)
    L = e.dot(e)
    
    return L/N  



def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and loss
      
    N=len(y)
    e = y-tx.dot(w)
    del_L=np.transpose(tx).dot(e)
    
    return -del_L/N
    # ***************************************************
    #raise NotImplementedError


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        #raise NotImplementedError
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        
        w = w - gamma*gradient
        
        # ***************************************************
        #raise NotImplementedError
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    
    rand_sample=np.random.randint(len(y))
    e = y[rand_sample]-tx[rand_sample].dot(w)
    del_L=np.transpose(tx[rand_sample]).dot(e)
    
    return -del_L
    
    # ***************************************************
    
    
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        gradient = compute_stoch_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        #raise NotImplementedError
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        
        w = w - gamma*gradient
        
        # ***************************************************
        #raise NotImplementedError
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1],


def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    N = len(y)
    w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    e=y-tx.dot(w)
    MSE = (e.dot(e))/N
    
    return w, MSE


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    
    N = len(y)
    w_ridge = np.linalg.inv(tx.T.dot(tx)+2*N*lambda_*np.identity(len(tx.T.dot(tx)))).dot(tx.T).dot(y)
    #w = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)
    e=y-tx.dot(w_ridge)
    MSE_ridge = (e.dot(e))/N + lambda_*np.linalg.norm(w_ridge)**2
    
    return w_ridge, MSE_ridge 


def sigmoid(t):
    """apply sigmoid function on t."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO

    sigma = np.exp(t)/(1+np.exp(t))
    sigma[sigma == np.nan] = 0.999
    sigma[sigma == 1] = 0.99
    sigma[sigma == 0] = 0.0001
    
    #np.where(sigma=='nan', 1, sigma)
    return sigma


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    #print(y.shape, tx.shape)
    S1 = y.T.dot(np.log(sigmoid(tx.dot(w))))
    S2 = (1-y).T.dot(np.log(1-sigmoid(tx.dot(w))))
    
    return -S1-S2

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    #print(y.shape, tx.shape, w.shape, tx.dot(w).shape)
    A = np.reshape(sigmoid(tx.dot(w)),[len(y),1])
    y = np.reshape(y, [len(y),1])
    #print(A.shape, y.shape, (A-np.reshape(y, [len(y),1])).shape)
    gradient = tx.T.dot(A-y)
    
    
    return gradient

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # compute the cost: TODO
    
    loss = calculate_loss(y, tx, w)
    
    # ***************************************************
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # compute the gradient: TODO
    
    grad = calculate_gradient(y, tx, w)
    
    # ***************************************************
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    #print(w.shape, grad.shape)
    w = w - gamma*grad
    #print(w.shape)
    # ***************************************************
    #raise NotImplementedError
    return w, loss

def logistic_regression(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.01
    losses = []
    ws = []
       
    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))
    
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        #print(w.shape)
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        ws.append(w)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        
    index_min = np.argmin(losses)

        
    return  ws[index_min], losses[index_min]
    


