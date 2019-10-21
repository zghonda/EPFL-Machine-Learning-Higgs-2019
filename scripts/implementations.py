from costs import compute_mse
from gradient_descent import *
from helpers import *
from proj1_helpers import predict_labels


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    w = initial_w

    for n_iter in range(max_iters):
        for mini_batch_y, mini_batch_tx in batch_iter(y, tx, 1):
            gradient = compute_stoch_gradient(mini_batch_y, mini_batch_tx, w)
            w = w - gamma * gradient

    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """calculate the least squares solution."""
    w_star = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    loss = compute_mse(y, tx, w_star)
    return w_star, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    lambda_bis = lambda_ / (2 * len(y))
    fst_term = np.linalg.inv(tx.T @ tx + lambda_bis * np.identity(tx.shape[1]))
    snd_term = tx.T @ y
    w_star = fst_term @ snd_term

    loss = compute_mse(y, tx, w_star) + lambda_ * np.linalg.norm(w_star) ** 2
    return w_star, loss


def predict(x_train, y_train, x_test, y_test, lambda_, degree):
    """ predict result."""
    # perform polynomial feature expansion
    x_train_poly = build_poly(x_train, degree)
    x_test_poly = build_poly(x_test, degree)

    # apply regression
    w, _ = ridge_regression(y_train, x_train_poly, lambda_)  # to change for logistic regression

    # predict labels
    y_pred_train = predict_labels(w, x_train_poly)
    y_pred_test = predict_labels(w, x_test_poly)

    return y_pred_train, y_pred_test
