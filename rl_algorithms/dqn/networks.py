# -*- coding: utf-8 -*-
"""MLP module for dqn algorithms

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_algorithms.common.helper_functions import identity, numpy2floattensor
from rl_algorithms.common.networks.heads import MLP, init_layer_uniform
from rl_algorithms.dqn.linear import NoisyLinearConstructor, NoisyMLPHandler
from rl_algorithms.registry import HEADS
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
@HEADS.register_module
class DuelingMLP(MLP, NoisyMLPHandler):
    """Multilayer perceptron with dueling construction."""

    def __init__(
        self,
        configs: ConfigDict,
        hidden_activation: Callable = F.relu,
    ):
        """Initialize."""
        if configs.use_noisy_net:
            linear_layer = NoisyLinearConstructor(configs.std_init)
            init_fn: Callable = identity
        else:
            linear_layer = nn.Linear
            init_fn = init_layer_uniform
        super(DuelingMLP, self).__init__(
            configs=configs,
            hidden_activation=hidden_activation,
            linear_layer=linear_layer,
            use_output_layer=False,
        )
        in_size = configs.hidden_sizes[-1]

        # set advantage layer
        self.advantage_hidden_layer = self.linear_layer(in_size, in_size)
        self.advantage_layer = self.linear_layer(in_size, configs.output_size)
        self.advantage_layer = init_fn(self.advantage_layer)

        # set value layer
        self.value_hidden_layer = self.linear_layer(in_size, in_size)
        self.value_layer = self.linear_layer(in_size, 1)
        self.value_layer = init_fn(self.value_layer)

    def forward_(self, x: torch.Tensor) -> torch.Tensor:
        adv_x = self.hidden_activation(self.advantage_hidden_layer(x))
        val_x = self.hidden_activation(self.value_hidden_layer(x))

        advantage = self.advantage_layer(adv_x)
        value = self.value_layer(val_x)
        advantage_mean = advantage.mean(dim=-1, keepdim=True)

        q = value + advantage - advantage_mean

        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        x = super(DuelingMLP, self).forward(x)
        x = self.forward_(x)

        return x


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
@HEADS.register_module
class C51DuelingMLP(MLP, NoisyMLPHandler):
    """Multilayered perceptron for C51 with dueling construction."""

    def __init__(
        self,
        configs: ConfigDict,
        hidden_activation: Callable = F.relu,
    ):
        """Initialize."""
        if configs.use_noisy_net:
            linear_layer = NoisyLinearConstructor(configs.std_init)
            init_fn: Callable = identity
        else:
            linear_layer = nn.Linear
            init_fn = init_layer_uniform
        super(C51DuelingMLP, self).__init__(
            configs=configs,
            hidden_activation=hidden_activation,
            linear_layer=linear_layer,
            use_output_layer=False,
        )
        in_size = configs.hidden_sizes[-1]
        self.action_size = configs.output_size
        self.atom_size = configs.atom_size
        self.output_size = configs.output_size * configs.atom_size
        self.v_min, self.v_max = configs.v_min, configs.v_max

        # set advantage layer
        self.advantage_hidden_layer = self.linear_layer(in_size, in_size)
        self.advantage_layer = self.linear_layer(in_size, self.output_size)
        self.advantage_layer = init_fn(self.advantage_layer)

        # set value layer
        self.value_hidden_layer = self.linear_layer(in_size, in_size)
        self.value_layer = self.linear_layer(in_size, self.atom_size)
        self.value_layer = init_fn(self.value_layer)

    def forward_(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get distribution for atoms."""
        action_size, atom_size = self.action_size, self.atom_size

        x = super(C51DuelingMLP, self).forward(x)
        adv_x = self.hidden_activation(self.advantage_hidden_layer(x))
        val_x = self.hidden_activation(self.value_hidden_layer(x))

        advantage = self.advantage_layer(adv_x).view(-1, action_size, atom_size)
        value = self.value_layer(val_x).view(-1, 1, atom_size)
        advantage_mean = advantage.mean(dim=1, keepdim=True)

        q_atoms = value + advantage - advantage_mean
        dist = F.softmax(q_atoms, dim=2)

        support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(device)
        q = torch.sum(dist * support, dim=2)

        return dist, q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        _, q = self.forward_(x)

        return q


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
@HEADS.register_module
class IQNMLP(MLP, NoisyMLPHandler):
    """Multilayered perceptron for IQN with dueling construction.

    Reference: https://github.com/google/dopamine
    """

    def __init__(
        self,
        configs: ConfigDict,
        hidden_activation: Callable = F.relu,
    ):
        """Initialize."""
        if configs.use_noisy_net:
            linear_layer = NoisyLinearConstructor(configs.std_init)
            init_fn: Callable = identity
        else:
            linear_layer = nn.Linear
            init_fn = init_layer_uniform
        super(IQNMLP, self).__init__(
            configs=configs,
            hidden_activation=hidden_activation,
            linear_layer=linear_layer,
            init_fn=init_fn,
        )

        self.n_quantiles = configs.n_quantile_samples
        self.quantile_embedding_dim = configs.quantile_embedding_dim
        self.input_size = configs.input_size
        self.output_size = configs.output_size

        # set quantile_net layer
        self.quantile_fc_layer = self.linear_layer(
            self.quantile_embedding_dim, self.input_size
        )
        self.quantile_fc_layer = init_fn(self.quantile_fc_layer)

    def forward_(
        self, state: torch.Tensor, n_tau_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get quantile values and quantiles."""
        batch_size = np.prod(state.size()) // self.input_size

        state_tiled = state.repeat(n_tau_samples, 1)

        # torch.rand (CPU) may make a segmentation fault due to its non-thread safety.
        # on v0.4.1
        # check: https://bit.ly/2TXlNbq
        quantiles = np.random.rand(n_tau_samples * batch_size, 1)
        quantiles = numpy2floattensor(quantiles, torch.device("cpu"))
        quantile_net = quantiles.repeat(1, self.quantile_embedding_dim)
        quantile_net = (
            torch.arange(1, self.quantile_embedding_dim + 1, dtype=torch.float)
            * math.pi
            * quantile_net
        )
        quantile_net = torch.cos(quantile_net).to(device)
        quantile_net = F.relu(self.quantile_fc_layer(quantile_net))

        # Hadamard product
        quantile_net = state_tiled * quantile_net

        quantile_values = super(IQNMLP, self).forward(quantile_net)

        return quantile_values, quantiles.to(device)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        quantile_values, _ = self.forward_(state, self.n_quantiles)
        quantile_values = quantile_values.view(self.n_quantiles, -1, self.output_size)
        q = torch.mean(quantile_values, dim=0)

        return q
