# -*- coding: utf-8 -*-
"""DDPG with Behavior Cloning Algorithm.

This module demonstrates DDPG with Behavior Cloning model on the environment
with continuous action space in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1709.10089.pdf
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Behavior Cloning DDPG actor model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of the action value
        action_high (float): upper bound of the action value
        device (torch.device): device selection (cpu / gpu)

    Attributes:
        actor (nn.Sequential): actor model with FC layers
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of the action value
        action_high (float): upper bound of the action value
        device (torch.device): device selection (cpu / gpu)

    """

    def __init__(self, state_dim, action_dim,
                 action_low, action_high, device):
        """Initialization."""
        super(Actor, self).__init__()
        self.device = device

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
                        nn.Tanh()
                     )

    def forward(self, state):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            specific action

        """
        state = torch.tensor(state).float().to(self.device)
        action = self.actor(state)

        # adjust the output range to [action_low, action_high]
        scale_factor = (self.action_high - self.action_low) / 2
        reloc_factor = (self.action_high - scale_factor)
        action = action * scale_factor + reloc_factor
        action = torch.clamp(action, self.action_low, self.action_high)

        return action


class Critic(nn.Module):
    """Behavior Cloning DDPG critic model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        device (torch.device): device selection (cpu / gpu)

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        critic (nn.Sequential): critic model with FC layers
        device (torch.device): device selection (cpu / gpu)

    """

    def __init__(self, state_dim, action_dim, device):
        """Initialization."""
        super(Critic, self).__init__()
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.critic = nn.Sequential(
                        nn.Linear(self.state_dim+self.action_dim, 24),
                        nn.ReLU(),
                        nn.Linear(24, 48),
                        nn.ReLU(),
                        nn.Linear(48, 24),
                        nn.ReLU(),
                        nn.Linear(24, 1),
                     )

    def forward(self, state, action):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space
            action (torch.Tensor): input tensor on the action space

        Returns:
            predicted state value

        """
        state = torch.tensor(state).float().to(self.device)

        x = torch.cat((state, action), dim=-1)  # concat action
        predicted_value = self.critic(x)

        return predicted_value
