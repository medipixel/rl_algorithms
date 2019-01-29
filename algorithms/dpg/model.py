# -*- coding: utf-8 -*-
"""Deterministic Policy Gradient Algorithm.

This module demonstrates DPG on-policy model on the environment
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://proceedings.mlr.press/v32/silver14.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """DPG actor model with simple FC layers.

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        actor (nn.Sequential): actor model with FC layers

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
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
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
        action = self.actor(state)

        return action


class Critic(nn.Module):
    """DPG critic model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim: int, action_dim: int):
        """Initialization."""
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim + self.action_dim, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        Args:
            state (torch.Tensor): input vector on the state space

        Returns:
            predicted state value

        """
        x = torch.cat((state, action), dim=-1)  # concat action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        predicted_value = self.fc4(x)

        return predicted_value
