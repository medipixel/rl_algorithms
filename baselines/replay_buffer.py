# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object."""
        self.device = device
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for e in experiences:
            states.append(np.expand_dims(e[0], axis=0))
            actions.append(e[1])
            rewards.append(e[2])
            next_states.append(np.expand_dims(e[3], axis=0))
            dones.append(e[4])

        states =\
            torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions =\
            torch.from_numpy(np.vstack(actions)).float().to(self.device)
        rewards =\
            torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states =\
            torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones =\
            torch.from_numpy(
                np.vstack(dones).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def replace(self, memory):
        """
        Replace memory to already exist memory
        (i.e. demonstration memory)
        """
        self.memory = memory

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
