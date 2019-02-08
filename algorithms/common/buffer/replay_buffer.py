# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py

    Attributes:
        buffer (list): deque of replay buffer
        batch_size (int): size of a batched sampled from replay buffer for training

    """

    def __init__(self, buffer_size: int, batch_size: int, demo: deque = None):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            demo (deque) : demonstration deque

        """
        self.buffer = list() if not demo else list(demo)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.idx = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add a new experience to memory."""
        data = (state, action, reward, next_state, done)

        if len(self.buffer) == self.buffer_size:
            self.buffer[self.idx] = data
            self.idx = (self.idx + 1) % self.buffer_size
        else:
            self.buffer.append(data)

    def extend(self, transitions: list):
        """Add experiences to memory."""
        self.buffer.extend(transitions)

    def sample(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.buffer, k=self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for e in experiences:
            states.append(np.expand_dims(e[0], axis=0))
            actions.append(e[1])
            rewards.append(e[2])
            next_states.append(np.expand_dims(e[3], axis=0))
            dones.append(e[4])

        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)
