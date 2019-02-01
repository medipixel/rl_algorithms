# -*- coding: utf-8 -*-
"""MLP module for model of algorithms

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Reference: https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/networks.py
"""

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    """Baseline of Multilayer perceptron.

    Attributes:
        input_size (int): size of input
        output_sizes (list): sizes of output layers
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers

    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        output_activation: Callable = None,
    ):
        """Initialization.

        Args:
            input_size (int): size of input
            output_sizes (list): sizes of output layers
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer

        """
        super(MLP, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.hidden_layers: list = []
        self.output_layers: list = []
        in_size = self.input_size
        # set hidden layers
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size

            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        in_size = hidden_sizes[-1]
        for i, output_size in enumerate(output_sizes):
            output_layer = nn.Linear(in_size, output_size)

            self.__setattr__("output_fc{}".format(i), fc)
            self.output_layers.append(output_layer)

    def forward(self, x: torch.Tensor):
        """Forward method implementation."""
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        output_list = list()
        for output_layer in self.output_layers:
            x = output_layer(x)
            if self.output_activation:
                x = self.output_activation(x)
            output_list.append(x)
        return tuple(output_list)


class GaussianMLP(MLP):
    """Multilayer perceptron with Gaussian distribution output."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        output_activation: Callable = None,
    ):
        """Initialization."""
        super(GaussianMLP, self).__init__(
            input_size, output_sizes, hidden_sizes, hidden_activation, output_activation
        )
        assert len(self.output_layers) == 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation."""
        # hidden layer
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        # get mean, std, log_std
        mu = self.output_layers[0](x)
        log_std = torch.zeros_like(mu)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        action = torch.clamp(dist.rsample(), -1.0, 1.0)

        return action, dist


class GaussianTanhMLP(MLP):
    """Multilayer perceptron with Gaussian distribution output.

    Attributes:
            log_std_min (float): lower bound of log std
            log_std_max (float): upper bound of log std

    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        output_activation: Callable = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """Initialization.

        Args:
            log_std_min (float): lower bound of log std
            log_std_max (float): upper bound of log std

        """
        super(GaussianTanhMLP, self).__init__(
            input_size, output_sizes, hidden_sizes, hidden_activation, output_activation
        )
        assert len(self.output_layers) == 2
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(
        self, x: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        # hidden layer
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        # get mean, std, log_std
        mu = self.output_layers[0](x)
        log_std = torch.clamp(
            self.output_layers[1](x), self.log_std_min, self.log_std_max
        )
        std = torch.exp(log_std)

        # normalize action and log_prob
        # see appendix C of 'https://arxiv.org/pdf/1812.05905.pdf'
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mu, std
