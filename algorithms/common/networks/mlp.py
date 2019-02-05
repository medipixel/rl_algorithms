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

from algorithms.common.helper_functions import identity


class MLP(nn.Module):
    """Baseline of Multilayer perceptron.

    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        output_activation: Callable = identity,
        use_output_layer: bool = True,
        init_w: float = 3e-3,
    ):
        """Initialization.

        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer
            use_output_layer (bool): whether or not to use the last layer
            init_w (float): weight initialization bound for the last layer

        """
        super(MLP, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_output_layer = use_output_layer

        # set hidden layers
        self.hidden_layers: list = []
        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        if self.use_output_layer:
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
        assert self.use_output_layer

        x = self.get_last_activation(x)

        output = self.output_layer(x)
        output = self.output_activation(output)

        return output


class GaussianDist(MLP):
    """Multilayer perceptron with Gaussian distribution output.

    Attributes:
        mu_activation (function): bounding function for mean
        log_std_clamping (bool): whether or not to clamp log std
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
        log_std_min: float = -20,
        log_std_max: float = 2,
        init_w: float = 3e-3,
    ):
        """Initialization.

        """
        super(GaussianDist, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )

        self.mu_activation = mu_activation
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

    def get_dist_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return gausian distribution parameters."""
        hidden = super(GaussianDist, self).get_last_activation(x)

        # get mean
        mu = self.mu_activation(self.mu_layer(hidden))

        # get std
        log_std = torch.clamp(
            self.log_std_layer(hidden), self.log_std_min, self.log_std_max
        )
        std = torch.exp(log_std)

        return mu, log_std, std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        mu, _, std = self.get_dist_params(x)

        # get normal distribution and action
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class GaussianDistParams(GaussianDist):
    """Multilayer perceptron with Gaussian distribution params output."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        mu, log_std, std = super(GaussianDistParams, self).get_dist_params(x)

        return mu, log_std, std


class TanhGaussianDistParams(GaussianDist):
    """Multilayer perceptron with Gaussian distribution output."""

    def __init__(self, **kwargs):
        """Initialization."""
        super(TanhGaussianDistParams, self).__init__(**kwargs, mu_activation=identity)

    def forward(
        self, x: torch.Tensor, epsilon: float = 1e-6
    ) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        mu, _, std = super(TanhGaussianDistParams, self).get_dist_params(x)

        # sampling actions
        dist = Normal(mu, std)
        z = dist.rsample()

        # normalize action and log_prob
        # see appendix C of 'https://arxiv.org/pdf/1812.05905.pdf'
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mu, std
