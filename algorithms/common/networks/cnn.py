# -*- coding: utf-8 -*-
"""MLP module for model of algorithms

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

from typing import Callable

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from algorithms.common.helper_functions import identity

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class _CNNLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation_fn: Callable = F.relu,
    ):
        super(_CNNLayer, self).__init__()

        self.cnn = nn.Conv2d(
            input_size,
            output_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.activation_fn = activation_fn

    def forward(self, x):
        return self.activation_fn(self.cnn(x), inplace=True)


class CNN(nn.Module):
    """Baseline of Convolution neural network."""

    def __init__(
        self,
        input_shape: list,  # [channel, width, height]
        feature_sizes: list,
        kernel_sizes: list,
        strides: list,
        hidden_sizes: list,
        output_size: int,
        cnn_activation: Callable = F.relu,
        hidden_activation: Callable = F.relu,
        output_activation: Callable = identity,
    ):
        super(CNN, self).__init__()
        self.input_shape = input_shape
        self.feature_sizes = feature_sizes
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.cnn_activation = cnn_activation
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        in_channel = self.input_shape[0]
        self.cnn = nn.Sequential()
        for i in range(len(self.feature_sizes)):
            f, k, s = feature_sizes[i], kernel_sizes[i], strides[i]
            print(f, k, s)
            self.cnn.add_module(
                "cnn_{}".format(i),
                _CNNLayer(in_channel, f, k, s, activation_fn=self.cnn_activation),
            )
            in_channel = f

        cnn_output_size = self.feature_size()

        # set hidden layers
        self.hidden_layers: list = []
        in_size = cnn_output_size
        print(in_size)
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("hidden_fc{}".format(i), fc)
            self.hidden_layers.append(fc)

        # set output layers
        self.output_layer = nn.Linear(in_size, output_size)

    def feature_size(self):
        """Calculate flatten sizes of CNN output"""
        return (
            self.cnn(autograd.Variable(torch.zeros(1, *self.input_shape)))
            .view(1, -1)
            .size(1)
        )

    def get_last_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the activation of the last hidden layer."""
        x = self.cnn(x)
        x = x.view(x.size(0), -1)

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
