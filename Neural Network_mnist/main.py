# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        # raise NotImplementedError("Your Code Goes Here")
        self.h = h
        self.d = d
        self.k = k
        self.alpha_0 = 1 / np.sqrt(d)
        self.alpha_1 = 1 / np.sqrt(h)
        self.W0 = torch.FloatTensor(h, d).uniform_(-self.alpha_0, self.alpha_0)
        self.b0 = torch.FloatTensor(1, h).uniform_(-self.alpha_0, self.alpha_0)
        self.W1 = torch.FloatTensor(k, h).uniform_(-self.alpha_1, self.alpha_1)
        self.b1 = torch.FloatTensor(1, k).uniform_(-self.alpha_1, self.alpha_1)

        self.params = [self.W0, self.b0, self.W1, self.b1]
        for param in self.params:
            param.requires_grad = True

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        # raise NotImplementedError("Your Code Goes Here")
        x = torch.matmul(x, self.W0.T) + self.b0
        x = relu(x)
        x = torch.matmul(x, self.W1.T) + self.b1
        return x


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        # raise NotImplementedError("Your Code Goes Here")
        self.h0 = h0
        self.h1 = h1
        self.d = d
        self.k = k
        self.alpha_0 = 1 / np.sqrt(d)
        self.alpha_1 = 1 / np.sqrt(h0)
        self.alpha_2 = 1 / np.sqrt(h1)
        self.W0 = torch.FloatTensor(h0, d).uniform_(-self.alpha_0, self.alpha_0)
        self.b0 = torch.FloatTensor(1, h0).uniform_(-self.alpha_0, self.alpha_0)
        self.W1 = torch.FloatTensor(h1, h0).uniform_(-self.alpha_1, self.alpha_1)
        self.b1 = torch.FloatTensor(1, h1).uniform_(-self.alpha_1, self.alpha_1)
        self.W2 = torch.FloatTensor(k, h1).uniform_(-self.alpha_2, self.alpha_2)
        self.b2 = torch.FloatTensor(1, k).uniform_(-self.alpha_2, self.alpha_2)

        self.params = [self.W0, self.b0, self.W1, self.b1, self.W2, self.b2]
        for param in self.params:
            param.requires_grad = True

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: LongTensor of shape (n, k). Prediction.
        """
        # raise NotImplementedError("Your Code Goes Here")
        x = torch.matmul(x, self.W0.T) + self.b0
        x = relu(x)
        x = torch.matmul(x, self.W1.T) + self.b1
        x = relu(x)
        x = torch.matmul(x, self.W2.T) + self.b2
        return x


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    # raise NotImplementedError("Your Code Goes Here")
    losses = []
    for i in range(32):
        loss_epoch = 0
        acc = 0
        for batch in tqdm(train_loader):
            x, y = batch
            x = x.view(-1, 784)  # reshape to 784 x 1
            optimizer.zero_grad()
            logits = model.forward(x)
            preds = torch.argmax(logits, 1)
            acc += torch.sum(preds == y).item()
            loss = cross_entropy(logits, y)
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch ", i + 1)
        print("Loss:", loss_epoch / len(train_loader.dataset))
        print("Acc:", acc / len(train_loader.dataset))
        losses.append(loss_epoch / len(train_loader.dataset))
        if acc / len(train_loader.dataset) > 0.99:
            break
    return losses


def get_acc_test(test_loader, model):
    acc = 0
    loss_epoch = 0
    for batch in tqdm(test_loader):
        x, y = batch
        x = x.view(-1, 784)

        logits = model.forward(x)
        preds = torch.argmax(logits, 1)
        acc += torch.sum(preds == y).item()
        loss = cross_entropy(logits, y)
        loss_epoch += loss.item()

    l = loss_epoch / len(test_loader.dataset)
    a = acc / len(test_loader.dataset)
    return l, a


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    # raise NotImplementedError("Your Code Goes Here")
    train_dataset = TensorDataset(x, y)
    train_l = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_l = DataLoader(test_dataset, batch_size=128, shuffle=True)
    # F1 model
    M1 = F1(h=64, d=784, k=10)
    opt = Adam(M1.params, lr=0.001)
    l1 = train(M1, opt, train_l)
    # print(l)

    # plot for F1 model
    epo_time = range(1, len(l1) + 1)
    plt.plot(epo_time, l1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training Loss for F1 model')
    plt.show()

    # valid the test set
    ll, aa = get_acc_test(test_l, M1)
    print("Test dataset:")
    print("Loss:", ll)
    print("Acc:", aa)

    # number of parameter
    num_of_param_F1 = 0
    for p in M1.params:
        num_of_param_F1 += np.prod(p.shape)
    print("There are", num_of_param_F1, "trainable parameters in model for F1.")
    print("####################################################################")  # seperate line
    ###########################################################################
    # F2 model
    M2 = F2(h0=32, h1=32, d=784, k=10)
    opt = Adam(M2.params, lr=0.001)
    l2 = train(M2, opt, train_l)
    # print(l)

    # plot for F2 model
    epo_time = range(1, len(l2) + 1)
    plt.plot(epo_time, l2)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training Loss for F2 model')
    plt.show()

    # valid the test set
    ll, aa = get_acc_test(test_l, M2)
    print("Test dataset:")
    print("Loss:", ll)
    print("Acc:", aa)

    # number of parameter
    num_of_param_F2 = 0
    for p in M2.params:
        num_of_param_F2 += np.prod(p.shape)
    print("There are", num_of_param_F2, "trainable parameters in model for F2.")


if __name__ == "__main__":
    main()
