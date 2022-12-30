if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # raise NotImplementedError("Your Code Goes Here")
    y_train = df_train['ViolentCrimesPerPop'].to_numpy()
    y_test = df_test['ViolentCrimesPerPop'].to_numpy()
    X_train = df_train.drop('ViolentCrimesPerPop', axis=1).to_numpy()
    X_test = df_test.drop('ViolentCrimesPerPop', axis=1).to_numpy()

    lambda_max = np.max(np.sum(2 * X_train * (y_train - np.mean(y_train))[:, None], axis=0))
    print("Lambda_max is :", lambda_max)

    # generate a bunch of lambda
    lambdas = [lambda_max / (2**i) for i in range(17)]  # I tried to use 20 first and found that when i = 17, lambda will smaller than 0.01

    # list to store result
    non_0 = []
    path_w = []
    train_mse = []
    test_mse = []

    # function to get predict value and find the mse
    def predict(X, w, b):
        return X @ w + b

    def mse(x, y):
        return np.mean((x - y) ** 2)

    # for  question e
    train_w, bias = train(X_train, y_train, 30)
    print("Result for Qeustion (f):")
    print(train_w)
    # start testing
    print("=================")
    print("Result for Question (c), (d), (e):")
    for l in lambdas:
        train_w, bias = train(X_train, y_train, l)

        # for question c
        res_non_zero = np.sum(abs(train_w) > 1e-10)
        non_0.append(res_non_zero)
        # for question d
        path_w.append(np.copy(train_w))
        # for quesiton e
        train_mse.append(mse(predict(X_train, train_w, bias), y_train))
        test_mse.append(mse(predict(X_test, train_w, bias), y_test))

        print("finish lambda = ", l)

    # plot for c
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, non_0, '-o')
    plt.xscale('log')
    plt.title('# of nonzero count')
    plt.xlabel('lambda (log scale)')
    plt.ylabel('# of non-zero variables')
    plt.show()

    # plot for d
    plt.figure(figsize=(12, 8))
    path_w = np.array(path_w)
    col_names = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    # col_index = [X_train.columns.get_loc(name) for name in col_names]
    col_index = [3, 12, 7, 5, 1]
    for p, label in zip(path_w[:, col_index].T, col_names):
        plt.plot(lambdas, p, '-o', label=label)
    plt.legend()
    plt.xscale('log')
    plt.title('Path for 5 variables')
    plt.xlabel('lambda (log scale)')
    plt.ylabel('weight')
    plt.show()

    # plot for e
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, train_mse, '-o', label='train_mse')
    plt.plot(lambdas, test_mse, '-o', label='test_mse')
    plt.xscale('log')
    plt.title('MSE for train data and test data')
    plt.legend()
    plt.xlabel('lambda (log scale)')
    plt.ylabel('Mean Square Error')
    plt.show()


if __name__ == "__main__":
    main()
