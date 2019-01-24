# -*- coding: utf-8 -*-
"""Reinforce method for episodic tasks with continuous actions in OpenAI Gym.

This module demonstrates Reinforce with baseline model on the episodic tasks
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Reinforce continuous actor-critic model with simple FC layers.

    Attributes:
        actor_std (float): standard deviation of the output distribution
        actor_mu (nn.Sequential): actor model for mu with FC layers
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of the action value
        action_high (float): upper bound of the action value
        actor_mu (nn.Sequential): actor model for mu with FC layers
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(
        self,
        std: float,
        state_dim: int,
        action_dim: int,
        action_low: float,
        action_high: float,
    ):
        """Initialization.

        Args:
            std (float): standard deviation of the output distribution
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space
            action_low (float): lower bound of the action value
            action_high (float): upper bound of the action value

        """
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        self.actor_std = std
        self.actor_mu = nn.Sequential(
            nn.Linear(self.state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_dim),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward method implementation.

        The original paper suggests to employ an approximator
        for standard deviation, but, practically, it shows worse performance
        rather than using constant value by setting a hyper-parameter.
        The default std is 1.0 that leads to a good result on
        LunarLanderContinuous-v2 environment.

        Args:
            state (torch.Tensor): input vector on the state space

        Returns:
            normal distribution parameters as the output of actor model and
            approximated value of the input state as the output of
            critic model

        """
        norm_dist_mu = self.actor_mu(state)
        norm_dist_std = self.actor_std
        predicted_value = self.critic(state)

        dist = Normal(norm_dist_mu, norm_dist_std)
        selected_action = torch.clamp(dist.rsample(), self.action_low, self.action_high)

        return selected_action, predicted_value, dist
