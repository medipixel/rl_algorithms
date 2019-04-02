# -*- coding: utf-8 -*-
"""MLP module for dqn algorithms

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import math
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common.networks.cnn import CNN
from algorithms.common.networks.mlp import MLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DuelingMLP(MLP):
    """Multilayer perceptron with dueling construction."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        init_w: float = 3e-3,
    ):
        """Initialization."""
        super(DuelingMLP, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )
        in_size = hidden_sizes[-1]

        # set advantage layer
        self.advantage_hidden_layer = nn.Linear(in_size, in_size)
        self.advantage_layer = nn.Linear(in_size, output_size)
        self.advantage_layer.weight.data.uniform_(-init_w, init_w)
        self.advantage_layer.bias.data.uniform_(-init_w, init_w)

        # set value layer
        self.value_hidden_layer = nn.Linear(in_size, in_size)
        self.value_layer = nn.Linear(in_size, 1)
        self.value_layer.weight.data.uniform_(-init_w, init_w)
        self.value_layer.bias.data.uniform_(-init_w, init_w)

    def _forward_dueling(self, x: torch.Tensor) -> torch.Tensor:
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
        x = self._forward_dueling(x)

        return x


class C51CNN(CNN):
    """Convolution neural network for c51."""

    def get_dist_q(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method implementation."""
        x = self.get_cnn_features(x)
        dist, q = self.fc_layers.get_dist_q(x)
        return dist, q


class C51DuelingMLP(MLP):
    """Multilayered perceptron for C51 with dueling construction."""

    def __init__(
        self,
        input_size: int,
        action_size: int,
        hidden_sizes: list,
        atom_size: int = 51,
        v_min: int = -10,
        v_max: int = 10,
        hidden_activation: Callable = F.relu,
        init_w: float = 3e-3,
    ):
        """Initialization."""
        super(C51DuelingMLP, self).__init__(
            input_size=input_size,
            output_size=action_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )
        in_size = hidden_sizes[-1]
        self.action_size = action_size
        self.atom_size = atom_size
        self.output_size = action_size * atom_size
        self.v_min, self.v_max = v_min, v_max

        # set advantage layer
        self.advantage_hidden_layer = nn.Linear(in_size, in_size)
        self.advantage_layer = nn.Linear(in_size, self.output_size)
        self.advantage_layer.weight.data.uniform_(-init_w, init_w)
        self.advantage_layer.bias.data.uniform_(-init_w, init_w)

        # set value layer
        self.value_hidden_layer = nn.Linear(in_size, in_size)
        self.value_layer = nn.Linear(in_size, self.atom_size)
        self.value_layer.weight.data.uniform_(-init_w, init_w)
        self.value_layer.bias.data.uniform_(-init_w, init_w)

    def get_dist_q(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        _, q = self.get_dist_q(x)

        return q


class IQNDuelingMLP(MLP):
    """Multilayered perceptron for IQN with dueling construction.

    Reference: https://github.com/google/dopamine
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        n_quantiles: int,
        quantile_embedding_dim: int,
        hidden_activation: Callable = F.relu,
        init_w: float = 3e-3,
    ):
        """Initialization."""
        super(IQNDuelingMLP, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )
        self.quantile_embedding_dim = quantile_embedding_dim
        self.input_size = input_size
        self.output_size = output_size
        self.n_quantiles = n_quantiles

        # set quantile_net layer
        self.quantile_fc_layer = nn.Linear(self.quantile_embedding_dim, self.input_size)
        self.quantile_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.quantile_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        batch_size = np.prod(state.size()) // self.input_size
        state_tiled = state.repeat(self.n_quantiles, 1)

        quantiles = torch.rand([self.n_quantiles * batch_size, 1])
        quantile_net = quantiles.repeat(1, self.quantile_embedding_dim)
        quantile_net = (
            torch.arange(1, self.quantile_embedding_dim + 1, dtype=torch.float)
            * math.pi
            * quantile_net
        )
        quantile_net = torch.cos(quantile_net)
        quantile_net = F.relu(self.quantile_fc_layer(quantile_net))

        # Hadamard product
        net = state_tiled * quantile_net

        quantile_values = super(IQNDuelingMLP, self).forward(net)
        quantile_values = quantile_values.view(
            self.n_quantiles, batch_size, self.output_size
        )
        q = torch.mean(quantile_values, dim=0)

        return q
