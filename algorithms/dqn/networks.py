# -*- coding: utf-8 -*-
"""MLP module for dqn algorithms

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

from typing import Any, Callable

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

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward method implementation."""
        x: torch.Tensor = args[0]
        epsilon: float = args[1]

        x = super(DuelingMLP, self).forward(x)

        adv_x = self.advantage_hidden_layer(x)
        val_x = self.value_hidden_layer(x)

        advantage = self.advantage_layer(adv_x)
        value = self.value_layer(val_x)

        advantage_expectation = (
            (1 - epsilon) * (epsilon / self.output_size) * advantage.max()
        )

        advantage_expectation += (epsilon / self.output_size) * (
            advantage.sum() - advantage.max()
        )

        return value + advantage - advantage_expectation


class DuelingCNN(CNN):
    """Dueling Convolution neural network."""

    def forward(self, *args: Any) -> torch.Tensor:
        """Forward method implementation."""
        x: torch.Tensor = args[0]
        epsilon: float = args[1]

        if len(x.size()) == 3:
            x = x.unsqueeze(0)

        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x, epsilon)
        return x
