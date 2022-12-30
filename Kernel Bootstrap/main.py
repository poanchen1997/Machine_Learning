from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem
# use for fix the random result of SGD
RNG = np.random.RandomState(seed=446)


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    # raise NotImplementedError("Your Code Goes Here")
    return np.power((np.multiply.outer(x_i, x_j) + 1), d)


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    # raise NotImplementedError("Your Code Goes Here")
    return np.exp(-gamma * np.power(np.subtract.outer(x_i, x_j), 2))


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    # raise NotImplementedError("Your Code Goes Here")
    x_bar = np.mean(x)
    x_std = np.std(x)
    x = (x - x_bar) / x_std

    K = kernel_function(x, x, kernel_param)
    return np.linalg.solve(K + _lambda * np.eye(K.shape[0]), y)


def predict(x_train, x_val, alpha, kernel_function, kernel_param):
    x_bar = np.mean(x_train)
    x_std = np.std(x_train)
    x_train = (x_train - x_bar) / x_std
    x_val = (x_val - x_bar) / x_std
    return alpha @ kernel_function(x_train, x_val, kernel_param)


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    # raise NotImplementedError("Your Code Goes Here")
    folds = RNG.choice((list(np.arange(num_folds)) * (fold_size))[:len(x)], size=len(x), replace=False)
    mse = np.zeros(num_folds)

    for j in range(num_folds):
        idx = (folds != j)  # data for training
        alpha = train(x[idx], y[idx], kernel_function, kernel_param, _lambda)
        y_val_pred = predict(x[idx], x[~idx], alpha, kernel_function, kernel_param)
        mse[j] = np.mean((y[~idx] - y_val_pred) ** 2)
    # print(mse)
    return np.mean(mse)


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    # raise NotImplementedError("Your Code Goes Here")
    lambdas = ([10 ** -i for i in range(1, 6)])
    gamma = 1 / np.median(np.subtract.outer(x, x) ** 2)
    best_err = np.inf
    best_l = None
    best_g = None

    for l in lambdas:
        mse = cross_validation(x, y, rbf_kernel, gamma, l, num_folds)
        # print(mse)
        if mse < best_err:
            best_l = l
            best_g = gamma
            best_err = mse
    return best_l, best_g


@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """
    # raise NotImplementedError("Your Code Goes Here")
    lambdas = ([10 ** -i for i in range(1, 6)])
    ds = [i for i in range(7, 22)]
    best_err = np.inf
    best_l = None
    best_d = None

    for l in lambdas:
        for d in ds:
            mse = cross_validation(x, y, poly_kernel, d, l, num_folds)
            # print(mse)
            if mse < best_err:
                best_l = l
                best_d = d
                best_err = mse
    return best_l, best_d


@problem.tag("hw3-A", start_line=1)
def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    bootstrap_iters: int = 300,
) -> np.ndarray:
    """Bootstrap function simulation empirical confidence interval of function class.

    For each iteration of bootstrap:
        1. Sample len(x) many of (x, y) pairs with replacement
        2. Train model on these sampled points
        3. Predict values on x_fine_grid (see provided code)

    Lastly after all iterations, calculated 5th and 95th percentiles of predictions for each point in x_fine_point and return them.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        bootstrap_iters (int, optional): [description]. Defaults to 300.

    Returns:
        np.ndarray: A (2, 100) numpy array, where each row contains 5 and 95 percentile of function prediction at corresponding point of x_fine_grid.

    Note:
        - See np.percentile function.
            It can take two percentiles at the same time, and take percentiles along specific axis.
    """
    x_fine_grid = np.linspace(0, 1, 100)
    # raise NotImplementedError("Your Code Goes Here")
    indices = np.arange(x.shape[0])  # choose from n indices
    y_val_pred_matrix = []

    for _ in range(bootstrap_iters):
        bootstrap_idx = np.random.choice(indices, size=x.shape[0], replace=True)
        x_boot, y_boot = x[bootstrap_idx], y[bootstrap_idx]
        alpha = train(x_boot, y_boot, kernel_function, kernel_param, _lambda)
        y_val_pred_matrix.append([predict(x_boot, x_fine_grid, alpha, kernel_function, kernel_param)])
    ci_lower = np.percentile(y_val_pred_matrix, 5, axis=0)
    ci_upper = np.percentile(y_val_pred_matrix, 95, axis=0)
    return np.concatenate((ci_lower, ci_upper), axis=0)


@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. For both rbf and poly kernels, plot 5th and 95th percentiles from bootstrap using x_30, y_30 (using the same fine grid as in part B)
        D. Repeat A, B, C with x_300, y_300
        E. Compare rbf and poly kernels using bootstrap as described in the pdf. Report 5 and 95 percentiles in errors of each function.

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    # raise NotImplementedError("Your Code Goes Here")

    # question a (n = 30)
    # poly kernel
    poly_l, poly_d = poly_param_search(x_30, y_30, 30)  # useing fold = n to implement LOO CV
    print(f"The optimal lambda and degree for polynomial kernel are {poly_l}, {poly_d}.")  # 0.001, 10
    # rbf kerenl
    rbf_l, rbf_g = rbf_param_search(x_30, y_30, 30)
    print(f"The optimal lambda and gamma for rbf kernel are {rbf_l}, {rbf_g}.")  # 0.1, 11.201924992299844

    # question b (n = 30)
    # same data for true
    x_true = np.linspace(0, 1, 100)
    y_true = f_true(x_true)

    # poly kernel using x_30 to train
    a_poly = train(x_30, y_30, poly_kernel, 10, 0.001)
    y_pred_poly = predict(x_30, sorted(x_30), a_poly, poly_kernel, 10)

    # plot for poly kernel
    plt.scatter(x_30, y_30)  # data point
    plt.plot(sorted(x_30), y_pred_poly, label="poly_predict", color='blue')  # poly predict
    plt.plot(x_true, y_true, label="true", color='green')             # true
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('plot for n = 30 with polynomial kerenl')
    plt.show()

    # rbf kernel using x_30 to train
    a_rbf = train(x_30, y_30, rbf_kernel, 11.2, 0.1)
    y_pred_rbf = predict(x_30, sorted(x_30), a_rbf, rbf_kernel, 11.2)
    # plot for rbf kernel
    plt.scatter(x_30, y_30)  # data point
    plt.plot(sorted(x_30), y_pred_rbf, label="rbf_predict", color='red')   # rbf predict
    plt.plot(x_true, y_true, label="true", color='green')             # true
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('plot for n = 30 with rbf kerenl')
    plt.show()

    # question c (n = 300)
    # poly kernel
    poly_l, poly_d = poly_param_search(x_300, y_300, 10)
    print(f"The optimal lambda and degree for polynomial kernel are {poly_l}, {poly_d}.")  # 0.1, 14
    # rbf kerenl
    rbf_l, rbf_g = rbf_param_search(x_300, y_300, 10)
    print(f"The optimal lambda and gamma for rbf kernel are {rbf_l}, {rbf_g}.")  # 0.1, 12.865211223327618

    # question c-b (n = 300)
    # same data for true
    x_true = np.linspace(0, 1, 100)
    y_true = f_true(x_true)

    # poly kernel using x_300 to train
    a_poly = train(x_300, y_300, poly_kernel, 14, 0.1)
    y_pred_poly = predict(x_300, sorted(x_300), a_poly, poly_kernel, 14)

    # plot for poly kernel
    plt.scatter(x_300, y_300)  # data point
    plt.plot(sorted(x_300), y_pred_poly, label="poly_predict", color='blue')  # poly predict
    plt.plot(x_true, y_true, label="true", color='green')             # true
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('plot for n = 300 with polynomial kerenl')
    plt.show()

    # rbf kernel using x_300 to train
    a_rbf = train(x_300, y_300, rbf_kernel, 12.87, 0.1)
    y_pred_rbf = predict(x_300, sorted(x_300), a_rbf, rbf_kernel, 12.87)
    # plot for rbf kernel
    plt.scatter(x_300, y_300)  # data point
    plt.plot(sorted(x_300), y_pred_rbf, label="rbf_predict", color='red')   # rbf predict
    plt.plot(x_true, y_true, label="true", color='green')             # true
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('plot for n = 300 with rbf kerenl')
    plt.show()

    # a = bootstrap(x_30, y_30, rbf_kernel, 11.2, 0.1, 300)
    # alpha = train(x_30, y_30, rbf_kernel, 11.2, 0.1)
    # y_pre = predict(x_30, sorted(x_30), alpha, rbf_kernel, 11.2)
    # x_plot = np.linspace(0, 1, 100)
    # plt.plot(sorted(x_30), y_pre)
    # plt.scatter(x_30, y_30)
    # plt.plot(x_plot, a[0])
    # plt.plot(x_plot, a[1])
    # plt.show()


if __name__ == "__main__":
    main()
