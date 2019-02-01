# -*- coding: utf-8 -*-
"""Trust Region Policy Optimization Algorithm.

This module demonstrates TRPO model on the environment
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://arxiv.org/abs/1502.05477
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """TRPO actor model with simple FC layers.

    Attributes:
        actor (nn.Sequential): actor model with FC layers
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        """Initialization.
        Args:
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space
            hidden_size (int): hidden layer size
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.mu = nn.Linear(hidden_size, self.action_dim)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.0)
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        Args:
            state (torch.Tensor): input vector on the state space

        Returns:
            specific action

        """
        hidden = self.hidden(state)
        mu = self.mu(hidden)
        std = self.log_std.exp().expand_as(mu)

        return mu, self.log_std, std


class Critic(nn.Module):
    """TRPO critic model with simple FC layers.

    Attributes:
        state_dim (int): dimension of state space
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim: int, hidden_size: int = 256, init_w: float = 3e-3):
        """Initialization.
        Args:
            state_dim (int): dimension of state space
            hidden_size (int): hidden layer size
            init_w (float): initial weight value of the last layer
        """
        super(Critic, self).__init__()

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
            predicted value for the input state
        """
        predicted_value = self.last_layer(self.hidden_layers(state))

        return predicted_value
