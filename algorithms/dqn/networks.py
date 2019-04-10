# -*- coding: utf-8 -*-
"""MLP module for dqn algorithms

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common.networks.cnn import CNN
from algorithms.common.networks.mlp import MLP, init_layer_uniform
from algorithms.dqn.linear import NoisyMLPHandler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DuelingMLP(MLP, NoisyMLPHandler):
    """Multilayer perceptron with dueling construction."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list,
        hidden_activation: Callable = F.relu,
        linear_layer: nn.Module = nn.Linear,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialization."""
        super(DuelingMLP, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            linear_layer=linear_layer,
            use_output_layer=False,
        )
        in_size = hidden_sizes[-1]

        # set advantage layer
        self.advantage_hidden_layer = self.linear_layer(in_size, in_size)
        self.advantage_layer = self.linear_layer(in_size, output_size)
        self.advantage_layer = init_fn(self.advantage_layer)

        # set value layer
        self.value_hidden_layer = self.linear_layer(in_size, in_size)
        self.value_layer = self.linear_layer(in_size, 1)
        self.value_layer = init_fn(self.value_layer)

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
    """Convolution neural network for distributional RL."""

    def forward_(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method implementation."""
        x = self.get_cnn_features(x)
        out = self.fc_layers.forward_(x)
        return out

    def reset_noise(self):
        """Re-sample noise for fc layers."""
        self.fc_layers.reset_noise()


class C51DuelingMLP(MLP, NoisyMLPHandler):
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
        linear_layer: nn.Module = nn.Linear,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialization."""
        super(C51DuelingMLP, self).__init__(
            input_size=input_size,
            output_size=action_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            linear_layer=linear_layer,
            use_output_layer=False,
        )
        in_size = hidden_sizes[-1]
        self.action_size = action_size
        self.atom_size = atom_size
        self.output_size = action_size * atom_size
        self.v_min, self.v_max = v_min, v_max

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


class IQNCNN(CNN):
    """Convolution neural network for distributional RL."""

    def forward_(
        self, x: torch.Tensor, n_tau_samples: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method implementation."""
        x = self.get_cnn_features(x)
        out = self.fc_layers.forward_(x, n_tau_samples)
        return out

    def reset_noise(self):
        """Re-sample noise for fc layers."""
        self.fc_layers.reset_noise()


class IQNMLP(MLP, NoisyMLPHandler):
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
        linear_layer: nn.Module = nn.Linear,
        init_fn: Callable = init_layer_uniform,
    ):
        """Initialization."""
        super(IQNMLP, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            linear_layer=linear_layer,
            init_fn=init_fn,
        )

        IQNMLP.n_quantiles = n_quantiles
        self.quantile_embedding_dim = quantile_embedding_dim
        self.input_size = input_size
        self.output_size = output_size

        # set quantile_net layer
        self.quantile_fc_layer = self.linear_layer(
            self.quantile_embedding_dim, self.input_size
        )
        self.quantile_fc_layer = init_fn(self.quantile_fc_layer)

    def forward_(
        self, state: torch.Tensor, n_tau_samples: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get quantile values and quantiles."""
        n_tau_samples = self.__get_n_tau_samples(n_tau_samples)
        batch_size = np.prod(state.size()) // self.input_size

        state_tiled = state.repeat(n_tau_samples, 1)

        # torch.rand (CPU) may make a segmentation fault due to its non-thread safety.
        # on v0.4.1
        # check: https://bit.ly/2TXlNbq
        quantiles = np.random.rand(n_tau_samples * batch_size, 1)
        quantiles = torch.FloatTensor(quantiles)
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

        return quantile_values, quantiles

    def forward(self, state: torch.Tensor, n_tau_samples: int = None) -> torch.Tensor:
        """Forward method implementation."""
        n_tau_samples = self.__get_n_tau_samples(n_tau_samples)

        quantile_values, _ = self.forward_(state, n_tau_samples)
        quantile_values = quantile_values.view(n_tau_samples, -1, self.output_size)
        q = torch.mean(quantile_values, dim=0)

        return q

    @staticmethod
    def __get_n_tau_samples(n_tau_samples: Optional[int]) -> int:
        """Get sample tau number."""
        if not n_tau_samples:
            return IQNMLP.n_quantiles
        else:
            return n_tau_samples
