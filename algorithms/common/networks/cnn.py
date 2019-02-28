# -*- coding: utf-8 -*-
"""CNN modules for RL algorithms.

- Authors: Kh Kim & Curt Park
- Contacts: kh.kim@medipixel.io
            curt.park@medipixel.io
"""

from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common.helper_functions import identity
from algorithms.common.networks.mlp import MLP, LateFusionMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNNLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation_fn: Callable = F.relu,
        pulling_fn: Callable = identity,
    ):
        super(CNNLayer, self).__init__()

        self.cnn = nn.Conv2d(
            input_size,
            output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.activation_fn = activation_fn
        self.pulling_fn = pulling_fn

    def forward(self, x):
        return self.pulling_fn(self.activation_fn(self.cnn(x)))


class CNN(nn.Module):
    """Baseline of Convolution neural network."""

    def __init__(self, cnn_layers: List[CNNLayer], fc_layers: MLP):
        super(CNN, self).__init__()

        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers

        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(self.cnn_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x).squeeze()
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class LateFusionCNN(nn.Module):
    """Convolution neural network with late fusion inputs."""

    def __init__(self, cnn_layers: List[CNNLayer], fc_layers: LateFusionMLP):
        super(LateFusionCNN, self).__init__()

        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers

        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(self.cnn_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

    def forward(self, x: torch.Tensor, late_in: list) -> torch.Tensor:
        """Forward method implementation."""
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x).squeeze()
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x, late_in)
        return x
