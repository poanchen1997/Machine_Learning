from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    # raise NotImplementedError("Your Code Goes Here")
    return demean_data @ uk @ uk.T


@problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    # raise NotImplementedError("Your Code Goes Here")
    return (np.linalg.norm(demean_data - reconstruct_demean(uk, demean_data)) ** 2) / demean_data.shape[0]


@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of it.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,). Should be in descending order.
            2. Matrix with eigenvectors as columns with shape (d, d)
    """
    # raise NotImplementedError("Your Code Goes Here")
    # values, vectors = np.linalg.eig(demean_data)
    # return sorted(values, reverse=True), vectors
    U, sigma, VT = np.linalg.svd(demean_data, False)
    n = demean_data.shape[0]
    V = VT.T
    return sigma ** 2 / n, V


@problem.tag("hw4-A", start_line=2)
def main():
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    If the handout instructs you to implement the following sub-problems, you should:

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues

    Part C:
        - Plot reconstruction error as a function of k (# of eigenvectors used)
            Use k from 1 to 101.
            Plot should have two lines, one for train, one for test.
        - Plot ratio of sum of eigenvalues remaining after k^th eigenvalue with respect to whole sum of eigenvalues.
            Use k from 1 to 101.

    Part D:
        - Visualize 10 first eigenvectors as 28x28 grayscale images.

    Part E:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.
    """
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")

    # raise NotImplementedError("Your Code Goes Here")
    mean = np.mean(x_tr, axis=0)
    demean_data = x_tr - mean

    values, vectors = calculate_eigen(demean_data)
    values_sum = np.sum(values)
    # part a
    print(f"Lambda_i for i = {1, 2, 10, 30, 50} is :{[values[i] for i in [0, 1, 9, 29, 49]]}")
    print(f"Sum of Lambda is {values_sum}.")

    # part c
    ks = [i for i in range(1, 102)]
    train_err = []
    test_err = []

    for k in ks:
        train_err.append(reconstruction_error(vectors[:, :k], demean_data))
        test_err.append(reconstruction_error(vectors[:, :k], x_test - mean))
    plt.figure()
    plt.plot(train_err, label="Trainning error")
    plt.plot(test_err, label="Testing error")
    plt.xlabel("K")
    plt.ylabel("reconstruction error")
    plt.title("K v.s. reconstruction error")
    plt.legend()
    plt.show()

    ratio = []
    for k in ks:
        ratio.append(1 - np.sum(values[:k]) / values_sum)

    plt.figure()
    plt.plot(ratio)
    plt.xlabel("K")
    plt.ylabel("Remaining ratio")
    plt.title("K v.s. Remaining ratio")
    plt.show()

    # part d
    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(vectors[:, i].reshape(28, 28), cmap='gray')
        ax.set_title(f'PCA Mode {i}')
    plt.tight_layout()
    plt.show()

    # part e --> question c in pdf
    def plot_for_reconstruction(digit_num):
        digit = x_tr[y_tr == digit_num][0]  # use the first image to test
        k_s = [5, 15, 40, 100]
        names = ['Original'] + [f"k = {k}" for k in k_s]
        r_d = [digit] + [reconstruct_demean(vectors[:, :k], digit - mean) + mean for k in k_s]

        fig, axes = plt.subplots(1, 5, figsize=(1.5 * 5, 2 * 1))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(r_d[i].reshape(28, 28), cmap='gray')
            ax.set_title(names[i])
        plt.tight_layout()
        plt.show()

    plot_for_reconstruction(2)
    plot_for_reconstruction(6)
    plot_for_reconstruction(7)

    # for question b in pdf
    # X = (X - \mu) @ V_k @ V_k^T + mu
    # where \mu is the mean of row
    # V_k is the eigenvector of the first K column of the V,
    # where V is a dXd matrix with each column is a eigen vector


if __name__ == "__main__":
    main()
