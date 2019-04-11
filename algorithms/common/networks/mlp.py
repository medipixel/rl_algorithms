# -*- coding: utf-8 -*-
"""MLP module for model of algorithms

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

from typing import Callable, Tuple

import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common.helper_functions import identity, make_one_hot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def concat(
    in_1: torch.Tensor, in_2: torch.Tensor, n_category: int = -1
) -> torch.Tensor:
    """Concatenate state and action tensors properly depending on the action."""
    in_2 = make_one_hot(in_2, n_category) if n_category > 0 else in_2

    if len(in_2.size()) == 1:
        in_2 = in_2.unsqueeze(0)

    in_concat = torch.cat((in_1, in_2), dim=-1)

    return in_concat


def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer"""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


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
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        output_activation: Callable = identity,
        linear_layer: nn.Module = nn.Linear,
        use_output_layer: bool = True,
        n_category: int = -1,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialization.

        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer
            linear_layer (nn.Module): linear layer of mlp
            use_output_layer (bool): whether or not to use the last layer
            n_category (int): category number (-1 if the action is continuous)
            init_fn (Callable): weight initialization function bound for the last layer

        """
        super(MLP, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category

        # set hidden layers
        self.hidden_layers: list = []
        in_size = self.input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = self.linear_layer(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, output_size)
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


class FlattenMLP(MLP):
    """Baseline of Multilayered perceptron for Flatten input."""

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        states, actions = args
        flat_inputs = concat(states, actions, self.n_category)
        return super(FlattenMLP, self).forward(flat_inputs)


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
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        mu_activation: Callable = torch.tanh,
        log_std_min: float = -20,
        log_std_max: float = 2,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialization."""
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
        self.log_std_layer = init_fn(self.log_std_layer)

        # set mean layer
        self.mu_layer = nn.Linear(in_size, output_size)
        self.mu_layer = init_fn(self.mu_layer)

    def get_dist_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Return gausian distribution parameters."""
        hidden = super(GaussianDist, self).forward(x)

        # get mean
        mu = self.mu_activation(self.mu_layer(hidden))

        # get std
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


class CategoricalDist(MLP):
    """Multilayer perceptron with categorial distribution output.

    Attributes:
        last_layer (nn.Linear): output layer for softmax
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialization."""
        super(CategoricalDist, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]

        # set log_std layer
        self.last_layer = nn.Linear(in_size, output_size)
        self.last_layer = init_fn(self.last_layer)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        hidden = super(CategoricalDist, self).forward(x)
        action_probs = F.softmax(self.last_layer(hidden), dim=-1)

        dist = Categorical(action_probs)
        selected_action = dist.sample()

        return selected_action, dist


class CategoricalDistParams(CategoricalDist):
    """Multilayer perceptron with Categorical distribution output."""

    def __init__(self, compatible_with_tanh_normal=False, **kwargs):
        """Initialization."""
        super(CategoricalDistParams, self).__init__(**kwargs)

        self.compatible_with_tanh_normal = compatible_with_tanh_normal

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward method implementation."""
        action, dist = super(CategoricalDistParams, self).forward(x)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        if self.compatible_with_tanh_normal:
            # in order to prevent from using the unavailable return values
            nan = float("nan")
            return action, log_prob, nan, nan, nan
        else:
            return action, log_prob
