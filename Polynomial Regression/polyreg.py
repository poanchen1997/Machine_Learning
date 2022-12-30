"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from statistics import mean
from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        self.normalize = None
        # You can add additional fields
        # raise NotImplementedError("Your Code Goes Here")

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        # raise NotImplemented Error("Your Code Goes Here")
        n = len(X)
        res = np.zeros(shape=(n, degree))

        for i in range(n):
            for j in range(degree):
                res[i][j] = X[i] ** (j + 1)

        return res
        # return X**np.arange(1, degree + 1)

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You need to apply polynomial expansion and scaling at first.
        """
        # raise NotImplementedError("Your Code Goes Here")
        n = len(X)
        # polynomial expansion
        X_ = self.polyfeatures(X, self.degree)
        # normalization (no normalization if n = 1)
        if n != 1:
            self.mean = X_.mean(axis=0)
            self.std = X_.std(axis=0)
        else:  # if mean = 0, std = 1, the value will not change after the normalization
            self.mean = 0
            self.std = 1

        X_ = (X_ - self.mean) / self.std
        # adding ones
        X_ = np.c_[np.ones([n, 1]), X_]

        # construct reg matrix
        reg_matrix = self.reg_lambda * np.eye(self.degree + 1)
        reg_matrix[0, 0] = 0  # don't change the left-up value, because it's constant

        # analytical solution (X'X + regMatrix)^-1 X' y
        self.weight = np.linalg.pinv(X_.T.dot(X_) + reg_matrix).dot(X_.T).dot(y)

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        # raise NotImplementedError("Your Code Goes Here")
        n = len(X)
        X_ = self.polyfeatures(X, self.degree)
        # normalize  # use the mean and std in fit data
        X_ = (X_ - self.mean) / self.std
        # adding ones
        X_ = np.c_[np.ones([n, 1]), X_]

        # predict
        return X_.dot(self.weight)


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    # raise NotImplementedError("Your Code Goes Here")
    n = len(a)
    return sum((a - b)**2) / n


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    # raise NotImplementedError("Your Code Goes Here")
    for i in range(1, n):  # start from 1 because I want it to converge in 1 sample
        Xtrain_i = Xtrain[0 : i + 1]
        Ytrain_i = Ytrain[0 : i + 1]

        model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(Xtrain_i, Ytrain_i)

        predictions_train_i = model.predict(Xtrain_i)
        predictions_test_i = model.predict(Xtest)

        errorTrain[i] = mean_squared_error(predictions_train_i, Ytrain_i)
        errorTest[i] = mean_squared_error(predictions_test_i, Ytest)
    return errorTrain, errorTest
