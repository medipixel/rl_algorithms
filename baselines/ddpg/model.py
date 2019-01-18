# -*- coding: utf-8 -*-
"""Deep Deterministic Policy Gradient Algorithm.

This module demonstrates DDPG model on the environment
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """DDPG actor model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of the action value
        action_high (float): upper bound of the action value
        device (str): device selection (cpu / gpu)

    Attributes:
        actor (nn.Sequential): actor model with FC layers
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of the action value
        action_high (float): upper bound of the action value
        device (str): device selection (cpu / gpu)

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
    """DDPG critic model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim, action_dim, device):
        """Initialization."""
        super(Critic, self).__init__()
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim+self.action_dim, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, 1)

    def forward(self, state, action):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            predicted state value

        """
        state = torch.tensor(state).float().to(self.device)

        x = torch.cat((state, action), dim=-1)  # concat action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        predicted_value = self.fc4(x)

        return predicted_value
