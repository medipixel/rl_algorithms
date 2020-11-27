# -*- coding: utf-8 -*-
"""Linear module for dqn algorithms

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_algorithms.common.helper_functions import numpy2floattensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    References:
        https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
        https://github.com/Kaixhin/Rainbow/blob/master/model.py

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialize."""
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = numpy2floattensor(np.random.normal(loc=0.0, scale=1.0, size=size), device)

        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )


class NoisyLinearConstructor:
    """Constructor class for changing hyper parameters of NoisyLinear.

    Attributes:
        std_init (float): initial std value

    """

    def __init__(self, std_init: float = 0.5):
        """Initialize."""
        self.std_init = std_init

    def __call__(self, in_features: int, out_features: int) -> NoisyLinear:
        """Return NoisyLinear instance set hyper parameters"""
        return NoisyLinear(in_features, out_features, self.std_init)


class NoisyMLPHandler:
    """Includes methods to handle noisy linear."""

    def reset_noise(self):
        """Re-sample noise"""
        for _, module in self.named_children():
            module.reset_noise()
