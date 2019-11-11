import numpy as np
def propagate(B, b, X, Y):
    """
    params:
    B: weights of size [1, X.shape[0]]
    b: bias
    X: matrix of observations and features size [X.shape[0], X.shape[1]]
    Y: matrix of actual observation size [Y.shape[0], 1]

    returns:
    grads: dict of gradients, dB of shape same as B and db of shape [1, 1].
    cost: MSE cost of shape [m, 1]
    """

    ## m is no of observations ie rows of X
    m = X.shape[0]

    # Calculate hypothesis
    y_hat = np.dot(X, B.T) + b

    y = Y.values.reshape(Y.shape[0], 1)

    # Compute Cost
    cost = (1 / (2 * m)) * np.sum((y - y_hat) ** 2)

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dB = (-1 / m) * np.dot((y - y_hat).T, X)

    db = -np.sum(y - y_hat) / m

    grads = {"dB": dB,
             "db": db}

    return grads, cost


def optimize(B, b, X, Y, num_iterations, learning_rate):
    """
    params:
    B: weights of size [1, X.shape[0]]
    b: bias
    X: matrix of observations and features size [X.shape[0], X.shape[1]]
    Y: matrix of actual observation size [Y.shape[0], 1]
    num_iterations: number of iterations
    learning_rate: learning rate
    returns:
    params: parameters B of shape [1, X.shape[0]] and bias
    grads: dict of gradients, dB of shape same as B and db
    costs:  MSE cost
    """
    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation call function propagate
        grads, cost = propagate(B, b, X, Y)

        # Retrieve derivatives from grads
        dB = grads["dB"]
        db = grads["db"]

        # update parameters
        B = B - learning_rate * dB
        b = b - learning_rate * db

        costs.append(cost)

    params = {"B": B,
              "b": b}

    grads = {"dB": dB,
             "db": db}

    return params, grads, costs


def predict(B, b, X):
    """:param
    B: weights
    b: bias
    X: matrix of observations and features
    """
  # Compute predictions for X
    Y_prediction = np.dot(X, B.T) + b
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5):
    """
    params:
    X_train: X_train
    Y_train: Y_train
    X_test: X_test
    Y_test: Y_test

    returns:
    d: dictionary
    """

    # initialize parameters with zeros
    B = np.zeros(shape=(1, X_train.shape[1]))
    b = 0

    # Gradient descent
    parameters, grads, costs = optimize(B, b, X_train, Y_train, num_iterations, learning_rate)

    # Retrieve parameters w and b from dictionary "parameters"
    B = parameters["B"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(B, b, X_test)
    Y_prediction_train = predict(B, b, X_train)

    Y_train = Y_train.values.reshape(Y_train.shape[0], 1)
    Y_test = Y_test.values.reshape(Y_test.shape[0], 1)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "B": B,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d