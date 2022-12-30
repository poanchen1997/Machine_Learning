from tkinter import N
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    # raise NotImplementedError("Your Code Goes Here")
    # n, d = X.shape
    # res = np.zeros(d)
    # for i in range(d):
    #     temp = 0
    #     for j in range(n):
    #         temp += X[j, i]**2
    #     res[i] = 2 * temp
    # return res
    return 2 * np.sum(X**2, axis=0)


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    # raise NotImplementedError("Your Code Goes Here")
    n, d = X.shape
    b = np.mean(y - (X @ weight))

    for i in range(d):
        not_k_columns = np.arange(d) != i
        a_k = a[i]
        c_k = 2 * np.sum(X[:, i] * (y - b - (X[:, not_k_columns] @ weight[not_k_columns])), axis=0)
        if c_k < -_lambda :
            weight[i] = (c_k + _lambda) / a_k
        elif c_k > _lambda:
            weight[i] = (c_k - _lambda) / a_k
        else:
            weight[i] = 0
    return weight, b


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    # raise NotImplementedError("Your Code Goes Here")
    first_part = (np.linalg.norm(X @ weight + bias - y)) ** 2
    second_part = _lambda * np.linalg.norm(weight, ord=1)
    return first_part + second_part


@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None
    # raise NotImplementedError("Your Code Goes Here")
    while not convergence_criterion(start_weight, old_w, convergence_delta):
        old_w = np.copy(start_weight)
        start_weight, bias = step(X, y, start_weight, a, _lambda)
        # print(start_weight, old_w)
    return start_weight, bias


@problem.tag("hw2-A")
def convergence_criterion(weight: np.ndarray, old_w: np.ndarray, convergence_delta: float) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    # raise NotImplementedError("Your Code Goes Here")
    if old_w is None:
        return False
    return (np.linalg.norm(weight - old_w, ord=np.inf) < convergence_delta)  # or (math.isclose(np.linalg.norm(weight - old_w, ord=np.inf), convergence_delta))


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    # raise NotImplementedError("Your Code Goes Here")
    n = 500
    d = 1000
    k = 100
    # suppose x are drown from normal(0, 1)
    X = np.random.normal(0, 1, (n, d))
    e = np.random.normal(0, 1, (n, ))
    # w generation
    w = np.zeros(d)
    for j in range(k):
        w[j] = (j + 1) / k
    # y generation
    y = X @ w + e
    # lambda start
    lambda_max = np.max(np.sum(2 * X * (y - np.mean(y))[:, None], axis=0))
    print("Lambda_max is :", lambda_max)

    # generate a bunch of lambda
    lambdas = [lambda_max / (2**i) for i in range(20)]

    # list to store result
    non_0 = []
    tpr = []
    fdr = []
    # start testing
    for l in lambdas:
        train_w, _ = train(X, y, l, 0.01)

        res_non_zero = np.sum(abs(train_w) > 1e-10)
        non_0.append(res_non_zero)
        res_incorrect = np.sum(train_w[abs(w) <= 1e-10] > 1e-10)
        res_correct = np.sum(train_w[abs(w) > 1e-10] > 1e-10)
        tpr.append(res_correct / k)
        fdr.append(res_incorrect / res_non_zero)
        print("finish lambda = ", l)

    # plot a
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, non_0, '-o')
    plt.xscale('log')
    plt.title('# of nonzero count')
    plt.xlabel('lambda (log scale)')
    plt.ylabel('# of non-zero variables')
    plt.show()

    # plot b
    plt.figure(figsize=(12, 8))
    plt.plot(fdr, tpr, '-o')
    plt.title('TPR v.s. FDR')
    plt.xlabel('False Discovery Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


if __name__ == "__main__":
    main()
