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


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    """TRPO actor model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        actor (nn.Sequential): actor model with FC layers

    """

    def __init__(self, state_dim, action_dim):
        """Initialization."""
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
                        nn.Tanh()
                     )

    def forward(self, state):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            specific action

        """
        state = torch.tensor(state).float().to(device)

        mu = self.actor(state)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)

        return mu, logstd, std


class Critic(nn.Module):
    """TRPO critic model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim, action_dim):
        """Initialization."""
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

    def forward(self, state):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            specific action

        """
        state = torch.tensor(state).float().to(device)
        predicted_value = self.critic(state)

        return predicted_value
