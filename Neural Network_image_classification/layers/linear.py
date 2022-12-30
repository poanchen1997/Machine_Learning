from typing import Optional

import torch
from torch import nn
import numpy as np

from utils import problem


class LinearLayer(nn.Module):
    @problem.tag("hw4-A", start_line=1)
    def __init__(
        self, dim_in: int, dim_out: int, generator: Optional[torch.Generator] = None
    ):
        """Linear Layer, which performs calculation of: x @ weight + bias

        In constructor you should initialize weight and bias according to dimensions provided.
        You should use torch.randn function to initialize them by normal distribution, and provide the generator if it's defined.

        Both weight and bias should be of torch's type float.
        Additionally, for optimizer to work properly you will want to wrap both weight and bias in nn.Parameter.

        Args:
            dim_in (int): Number of features in data input.
            dim_out (int): Number of features output data should have.
            generator (Optional[torch.Generator], optional): Generator to use when creating weight and bias.
                If defined it should be passed into torch.randn function.
                Defaults to None.

        Note:
            - YOU ARE NOT ALLOWED to use torch.nn.Linear (or it's functional counterparts) in this class
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        super().__init__()
        # raise NotImplementedError("Your Code Goes Here")
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = torch.randn(dim_in, dim_out, generator=generator)
        self.bias = torch.randn(dim_out, generator=generator)
        self.parameters = [self.weight, self.bias]
        for param in self.parameters:
            param.requires_grad = True

    @problem.tag("hw4-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actually perform multiplication x @ weight + bias

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with shape of (n, dim_in).
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, with shape of (n, dim_out).
                Output data.
        """
        # raise NotImplementedError("Your Code Goes Here")
        return torch.matmul(x.float(), self.weight) + self.bias
