# -*- coding: utf-8 -*-
"""Utility functions for DQN.

This module has DQN util functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
"""

from typing import Deque, List, Tuple

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_n_step_info(
    transitions: Deque, gamma: float
) -> Tuple[np.int64, np.ndarray, bool]:
    """Return n step reward, next state, and done."""
    assert transitions

    reward = 0
    next_state = transitions[-1][-2]  # next_state of the last transition
    done = transitions[-1][-1]  # done of the last transition

    for transition in reversed(transitions):
        _, _, r, n_s, d = transition

        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done


class NStepTransitionBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
        buffer (list): list of replay buffer
        buffer_size (int): buffer size not storing demos
        demo_size (int): size of a demo to permanently store in the buffer
        cursor (int): position to store next transition coming in

    """

    def __init__(self, buffer_size: int, demo: list = None):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            demo (list): demonstration transitions

        """
        assert buffer_size > 0

        self.buffer_size = buffer_size
        self.buffer: list = list()
        self.demo_size = 0
        self.cursor = 0

        # if demo exists
        if demo:
            self.demo_size = len(demo)
            self.buffer.extend(demo)

        self.buffer.extend([None] * self.buffer_size)

    def add(self, transition: Tuple[np.ndarray, ...]):
        """Add a new transition to memory."""
        idx = self.demo_size + self.cursor
        self.buffer[idx] = transition
        self.cursor = (self.cursor + 1) % self.buffer_size

    def extend(self, transitions: list):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(transition)

    def sample(self, indices: List[int]) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in np.nditer(indices):
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

        states_ = torch.FloatTensor(np.array(states)).to(device)
        actions_ = torch.FloatTensor(np.array(actions)).to(device)
        rewards_ = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states_ = torch.FloatTensor(np.array(next_states)).to(device)
        dones_ = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        if torch.cuda.is_available():
            states_ = states_.cuda(non_blocking=True)
            actions_ = actions_.cuda(non_blocking=True)
            rewards_ = rewards_.cuda(non_blocking=True)
            next_states_ = next_states_.cuda(non_blocking=True)
            dones_ = dones_.cuda(non_blocking=True)

        return states_, actions_, rewards_, next_states_, dones_
