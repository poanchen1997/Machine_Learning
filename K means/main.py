if __name__ == "__main__":
    from k_means import calculate_error, lloyd_algorithm  # type: ignore
else:
    from .k_means import lloyd_algorithm, calculate_error

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw4-A", start_line=1)
def main():
    """Main function of k-means problem

    If the handout instructs you to implement the following sub-problems, you should:
        a. Run Lloyd's Algorithm for k=10, and report 10 centers returned.
        b. For ks: 2, 4, 8, 16, 32, 64 run Lloyd's Algorithm,
            and report objective function value on both training set and test set.
            (All one plot, 2 lines)

    NOTE: This code takes a while to run. For debugging purposes you might want to change:
        x_train to x_train[:10000]. CHANGE IT BACK before submission.
    """
    (x_train, _), (x_test, _) = load_dataset("mnist")
    # raise NotImplementedError("Your Code Goes Here")

    # a part
    center_10 = lloyd_algorithm(x_train, 10)

    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(center_10[0][i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Centroid {i}')
    plt.show()
    # b part
    ks = [2, 4, 8, 16, 32, 64]
    # ks = [2, 5, 10, 20, 40, 80, 160, 320]
    train_set_error = []
    test_set_error = []

    for k in ks:
        print(f"Now using kernel as {k} to compute the error...")
        res = lloyd_algorithm(x_train, k)
        train_set_error.append(res[1][-1])
        test_set_error.append(calculate_error(x_test, res[0]))

    # print(train_set_error)
    # print(test_set_error)

    plt.figure()
    plt.plot(ks, train_set_error, label='Train')
    plt.plot(ks, test_set_error, label="Test")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
