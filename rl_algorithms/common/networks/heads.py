# -*- coding: utf-8 -*-
"""MLP module for model of algorithms

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

from typing import Callable, Tuple

import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.nn.functional as F

import rl_algorithms.common.helper_functions as helper_functions
from rl_algorithms.common.helper_functions import identity
from rl_algorithms.registry import HEADS
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer"""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
@HEADS.register_module
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
        n_category (int): category number (-1 if the action is continuous)

    """

    def __init__(
        self,
        configs: ConfigDict,
        hidden_activation: Callable = F.relu,
        linear_layer: nn.Module = nn.Linear,
        use_output_layer: bool = True,
        n_category: int = -1,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialize."""
        super(MLP, self).__init__()

        self.hidden_sizes = configs.hidden_sizes
        self.input_size = configs.input_size
        self.output_size = configs.output_size
        self.hidden_activation = (
            getattr(helper_functions, configs.hidden_activation)
            if "hidden_activation" in configs.keys()
            else hidden_activation
        )
        self.output_activation = getattr(helper_functions, configs.output_activation)
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category

        # set hidden layers
        self.hidden_layers: list = []
        in_size = self.input_size
        for i, next_size in enumerate(configs.hidden_sizes):
            fc = self.linear_layer(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, configs.output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))
        x = self.output_activation(self.output_layer(x))

        return x


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
@HEADS.register_module
class GaussianDist(MLP):
    """Multilayer perceptron with Gaussian distribution output.

    Attributes:
        mu_activation (function): bounding function for mean
        log_std_min (float): lower bound of log std
        log_std_max (float): upper bound of log std
        mu_layer (nn.Linear): output layer for mean
        log_std_layer (nn.Linear): output layer for log std
    """

    def __init__(
        self,
        configs: ConfigDict,
        hidden_activation: Callable = F.relu,
        mu_activation: Callable = torch.tanh,
        log_std_min: float = -20,
        log_std_max: float = 2,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialize."""
        super(GaussianDist, self).__init__(
            configs=configs,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )

        self.mu_activation = mu_activation
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        in_size = configs.hidden_sizes[-1]
        self.fixed_logstd = configs.fixed_logstd

        # set log_std
        if self.fixed_logstd:
            log_std = -0.5 * torch.ones(self.output_size, dtype=torch.float32)
            self.log_std = torch.nn.Parameter(log_std)
        else:
            self.log_std_layer = nn.Linear(in_size, configs.output_size)
            self.log_std_layer = init_fn(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(in_size, configs.output_size)
        self.mu_layer = init_fn(self.mu_layer)

    def get_dist_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return gausian distribution parameters."""
        hidden = super(GaussianDist, self).forward(x)

        # get mean
        mu = self.mu_activation(self.mu_layer(hidden))

        # get std
        if self.fixed_logstd:
            log_std = self.log_std
        else:
            log_std = torch.tanh(self.log_std_layer(hidden))
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
                log_std + 1
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


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
@HEADS.register_module
class TanhGaussianDistParams(GaussianDist):
    """Multilayer perceptron with Gaussian distribution output."""

    def __init__(self, **kwargs):
        """Initialize."""
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


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
@HEADS.register_module
class CategoricalDist(MLP):
    """Multilayer perceptron with Categorical distribution output."""

    def __init__(
        self,
        configs: ConfigDict,
        hidden_activation: Callable = F.relu,
    ):
        """Initialize."""
        super().__init__(
            configs=configs,
            hidden_activation=hidden_activation,
            use_output_layer=True,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        ac_logits = super().forward(x)

        # get categorical distribution and action
        dist = Categorical(logits=ac_logits)
        action = dist.sample()

        return action, dist
