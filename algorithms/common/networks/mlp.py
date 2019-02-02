# -*- coding: utf-8 -*-
"""MLP module for model of algorithms

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
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
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        output_activation: Callable = None,
        init_w: float = 3e-3,
    ):
        """Initialization.

        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer

        """
        super(MLP, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # set hidden layers
        self.hidden_layers: list = []
        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        self.output_layer = nn.Linear(in_size, output_size)
        self.output_layer.weight.data.uniform_(-init_w, init_w)
        self.output_layer.bias.data.uniform_(-init_w, init_w)

    def get_last_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the activation of the last hidden layer."""
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = self.get_last_activation(x)

        output = self.output_layer(x)
        if self.output_activation:
            output = self.output_activation(output)
        return output


class GaussianDistPolicy(MLP):
    """Multilayer perceptron with Gaussian distribution output.

    Attributes:
        mu_activation (function): bounding function for mean
        log_std_clamping (bool): wheather or not to clamp log std
        log_std_min (float): lower bound of log std
        log_std_max (float): upper bound of log std
        mu_layer (nn.Linear): output layer for mean
        log_std_layer (nn.Linear): output layer for log std
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        mu_activation: Callable = torch.tanh,
        log_std_clamping: bool = True,
        log_std_min: float = -20,
        log_std_max: float = 2,
        init_w: float = 3e-3,
    ):
        """Initialization.

        """
        super(GaussianDistPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )

        self.mu_activation = mu_activation
        self.log_std_clamping = log_std_clamping
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        in_size = hidden_sizes[-1]

        # set log_std layer
        self.log_std_layer = nn.Linear(in_size, output_size)
        self.log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.bias.data.uniform_(-init_w, init_w)

        # set mean layer
        self.mu_layer = nn.Linear(in_size, output_size)
        self.mu_layer.weight.data.uniform_(-init_w, init_w)
        self.mu_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        hidden = super(GaussianDistPolicy, self).get_last_activation(x)

        # get mean
        mu = self.mu_layer(hidden)
        if self.mu_activation:
            mu = self.mu_activation(mu)

        # get std
        log_std = self.log_std_layer(hidden)
        if self.log_std_clamping:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # get normal distribution and action
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class TanhGaussianParamsPolicy(MLP):
    """Multilayer perceptron with Gaussian distribution output.

    Attributes:
        log_std_clamping (bool): wheather or not to clamp log std
        log_std_min (float): lower bound of log std
        log_std_max (float): upper bound of log std
        mu_layer (nn.Linear): output layer for mean
        log_std_layer (nn.Linear): output layer for log std

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        log_std_clamping: bool = True,
        log_std_min: float = -20,
        log_std_max: float = 2,
        init_w: float = 3e-3,
    ):
        """Initialization.

        Args:
            init_w (float): initial range of log std layer
            log_std_min (float): lower bound of log std
            log_std_max (float): upper bound of log std

        """
        super(TanhGaussianParamsPolicy, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )
        self.log_std_clamping = log_std_clamping
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        in_size = hidden_sizes[-1]

        # set log_std layer
        self.log_std_layer = nn.Linear(in_size, output_size)
        self.log_std_layer.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer.bias.data.uniform_(-init_w, init_w)

        # set mean layer
        self.mu_layer = nn.Linear(in_size, output_size)
        self.mu_layer.weight.data.uniform_(-init_w, init_w)
        self.mu_layer.bias.data.uniform_(-init_w, init_w)

    def forward(
        self, x: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        hidden = super(TanhGaussianParamsPolicy, self).get_last_activation(x)

        # get mean
        mu = self.mu_layer(hidden)

        # get std
        log_std = self.log_std_layer(hidden)
        if self.log_std_clamping:
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # sampling actions
        dist = Normal(mu, std)
        z = dist.sample()

        # normalize action and log_prob
        # see appendix C of 'https://arxiv.org/pdf/1812.05905.pdf'
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mu, std
