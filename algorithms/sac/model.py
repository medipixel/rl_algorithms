# -*- coding: utf-8 -*-
"""SAC models.

This module defines SAC models on the environment
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
"""

import torch
import torch.nn as nn

from algorithms.distribution import TanhNormal


class Actor(nn.Module):
    """SAC actor model with simple FC layers.

    Attributes:
        hidden (nn.Sequential): common hidden layers
        mu (nn.Linear): layer for mean values
        log_std (nn.Linear): layer for log std values
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2,
        init_w: float = 3e-3,
    ):
        """Initialization.

        Args:
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space
            hidden_size (int): hidden layer size
            log_std_min (float): lower bound of log std
            log_std_max (float): upper bound of log std
            init_w (float): initial weight value of the last layers

        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.hidden = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, self.action_dim)
        self.log_std = nn.Linear(hidden_size, self.action_dim)

        # weight initialization
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.mu.bias.data.uniform_(-init_w, init_w)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        Args:
            state (torch.Tensor): input vector on the state space

        Returns:
            specific action and distribution

        """
        hidden = self.hidden(state)

        # get mean, std, log_std
        mu = self.mu(hidden)
        log_std = torch.clamp(self.log_std(hidden), self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # get dist, log_prob, pre_tanh_value
        dist = TanhNormal(mu, log_std.exp())
        action, pre_tanh_value = dist.rsample(return_pretanh_value=True)
        log_prob = dist.log_prob(action, pre_tanh_value=pre_tanh_value).sum(
            dim=-1, keepdim=True
        )

        return action, log_prob, pre_tanh_value, mu, std


class Qvalue(nn.Module):
    """SAC's Q function with simple FC layers.

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        init_w: float = 3e-3,
    ):
        """Initialization.

        Args:
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space
            hidden_size (int): hidden layer size
            init_w (float): initial weight value of the last layer

        """
        super(Qvalue, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden_layers = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.last_layer = nn.Linear(hidden_size, 1)

        # weight initialization
        self.last_layer.weight.data.uniform_(-init_w, init_w)
        self.last_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        Args:
            state (torch.Tensor): input vector on the state space
            action (torch.Tensor): input tensor on the action space

        Returns:
            predicted state value

        """
        x = torch.cat((state, action), dim=-1)  # concat action
        predicted_value = self.last_layer(self.hidden_layers(x))

        return predicted_value


class Value(nn.Module):
    """SAC's V function with simple FC layers.

    Attributes:
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim: int, hidden_size: int = 256, init_w: float = 3e-3):
        """Initialization.

        Args:
            state_dim (int): dimension of state space
            hidden_size (int): hidden layer size
            init_w (float): initial weight value of the last layer

        """
        super(Value, self).__init__()

        self.state_dim = state_dim

        self.hidden_layers = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.last_layer = nn.Linear(hidden_size, 1)

        # weight initialization
        self.last_layer.weight.data.uniform_(-init_w, init_w)
        self.last_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        Args:
            state (torch.Tensor): input vector on the state space

        Returns:
            predicted state value

        """
        predicted_value = self.last_layer(self.hidden_layers(state))

        return predicted_value
