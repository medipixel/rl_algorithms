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
from torch.distributions import Normal


class Actor(nn.Module):
    """TRPO actor model with simple FC layers.

    Attributes:
        actor (nn.Sequential): actor model with FC layers
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    """

    def __init__(self, state_dim: int, action_dim: int):
        """Initialization.

        Args:
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space

        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(self.state_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 48),
            nn.Tanh(),
            nn.Linear(48, 24),
            nn.Tanh(),
            nn.Linear(24, self.action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        Args:
            state (torch.Tensor): input vector on the state space

        Returns:
            specific action

        """
        mu = self.actor(state)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)

        dist = Normal(mu, std)
        selected_action = torch.clamp(dist.rsample(), -1.0, 1.0)

        return selected_action, dist


class Critic(nn.Module):
    """TRPO critic model with simple FC layers.

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
