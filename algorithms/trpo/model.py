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
                        nn.Tanh(),
                        nn.Linear(24, 48),
                        nn.Tanh(),
                        nn.Linear(48, 24),
                        nn.Tanh(),
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

        mu = self.actor(state)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)

        dist = Normal(mu, std)
        selected_action = torch.clamp(dist.rsample(),
                                      self.action_low,
                                      self.action_high)

        return selected_action, dist


class Critic(nn.Module):
    """TRPO critic model with simple FC layers.

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
                        nn.Linear(self.state_dim, 24),
                        nn.Tanh(),
                        nn.Linear(24, 48),
                        nn.Tanh(),
                        nn.Linear(48, 24),
                        nn.Tanh(),
                        nn.Linear(24, 1),
                     )

    def forward(self, state):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            specific action

        """
        state = torch.tensor(state).float().to(self.device)
        predicted_value = self.critic(state)

        return predicted_value
