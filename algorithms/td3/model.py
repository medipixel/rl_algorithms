# -*- coding: utf-8 -*-
"""TD3.

This module defines TD3 model on the environment
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1802.09477.pdf
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """TD3 actor model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of the action value
        action_high (float): upper bound of the action value

    Attributes:
        actor (nn.Sequential): actor model with FC layers
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of the action value
        action_high (float): upper bound of the action value

    """

    def __init__(
        self, state_dim: int, action_dim: int, action_low: float, action_high: float
    ):
        """Initialization."""
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

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

        # adjust the output range to [action_low, action_high]
        scale_factor = (self.action_high - self.action_low) / 2
        reloc_factor = self.action_high - scale_factor
        action = action * scale_factor + reloc_factor
        action = torch.clamp(action, self.action_low, self.action_high)

        return action


class Critic(nn.Module):
    """TD3 critic model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        device (torch.device): cpu or cuda

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

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space
            action (torch.Tensor): input tensor on the action space

        Returns:
            predicted state value

        """
        x = torch.cat((state, action), dim=-1)  # concat action
        predicted_value = self.critic(x)

        return predicted_value
