# -*- coding: utf-8 -*-
"""Proximal Policy Optimization Algorithms.

This module demonstrates PPO models on the environment
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """PPO actor model with simple FC layers.

    Attributes:
        hidden (nn.Sequential): hidden FC layers
        mu (nn.Linear): last layer for mean
        log_std (nn.Linear): last layer for log_std
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

        dist = Normal(mu, std)
        selected_action = dist.rsample()

        return selected_action, dist


class Critic(nn.Module):
    """PPO critic model with simple FC layers.

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim: int, action_dim: int):
        """Initialization.

        Args:
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space

        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 48),
            nn.Tanh(),
            nn.Linear(48, 24),
            nn.Tanh(),
            nn.Linear(24, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        Args:
            state (torch.Tensor): input vector on the state space

        Returns:
            predicted value for the input state

        """
        predicted_value = self.critic(state)

        return predicted_value
