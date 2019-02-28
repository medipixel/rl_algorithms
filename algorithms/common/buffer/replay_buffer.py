# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

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
        buffer (list): list of replay buffer
        batch_size (int): size of a batched sampled from replay buffer for training

    """

    def __init__(self, buffer_size: int, batch_size: int):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training

        """
        self.buffer: list = list()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.idx = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.ndarray,
        done: float,
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
        for transition in transitions:
            self.add(*transition)

    def sample(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in np.nditer(idxs):
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)
