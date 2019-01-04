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


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorCritic(nn.Module):
    """PPO actor-critic model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        actor (nn.Sequential): actor model with FC layers
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim, action_dim, action_low, action_high):
        """Initialization."""
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.actor = nn.Sequential(
                        nn.Linear(self.state_dim, 24),
                        nn.Tanh(),
                        nn.Linear(24, self.action_dim),
                        nn.Tanh()
                     )

        self.critic = nn.Sequential(
                        nn.Linear(self.state_dim, 24),
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

        mu = self.actor(state)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)

        dist = Normal(mu, std)
        selected_action = torch.clamp(dist.rsample(),
                                      self.action_low,
                                      self.action_high)

        predicted_value = self.critic(state)

        return selected_action, predicted_value, dist
