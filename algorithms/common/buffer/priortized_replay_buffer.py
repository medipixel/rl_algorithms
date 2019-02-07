# -*- coding: utf-8 -*-
"""Prioritized Replay buffer for baselines.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch

from algorithms.common.buffer.replay_buffer import ReplayBuffer
from algorithms.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer(ReplayBuffer):
    """Create Prioritized Replay buffer.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        buffer_size (int): size of replay buffer for experience
        alpha (float): alpha parameter for prioritized replay buffer
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        init_priority (float): lower bound of priority

        """

    def __init__(
        self, buffer_size: int, batch_size: int, demo: deque = None, alpha: float = 0.6
    ):
        """Initialization.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            demo (deque): demonstration
            alpha (float): alpha parameter for prioritized replay buffer

        """
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, batch_size, demo)
        assert alpha >= 0
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.init_priority = 1.0

        # for init priority of demo
        if demo:
            for _ in range(len(demo)):
                self.sum_tree[self.tree_idx] = self.init_priority ** self.alpha
                self.min_tree[self.tree_idx] = self.init_priority ** self.alpha
                self.tree_idx += 1

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience and priority."""
        idx = self.tree_idx
        self.tree_idx = (self.tree_idx + 1) % self.buffer_size
        super().add(state, action, reward, next_state, done)

        self.sum_tree[idx] = self.init_priority ** self.alpha
        self.min_tree[idx] = self.init_priority ** self.alpha

    def extend(self, transitions: list):
        """Add experiences to memory."""
        raise NotImplementedError

    def _sample_proportional(self, batch_size: int) -> list:
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self.buffer) - 1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def sample(self, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        assert beta > 0

        indices = self._sample_proportional(self.batch_size)
        states, actions, rewards, next_states, dones, weights = [], [], [], [], [], []

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        for idx in indices:
            s = self.buffer[idx]

            # append transition info
            states.append(np.expand_dims(s[0], axis=0))
            actions.append(s[1])
            rewards.append(s[2])
            next_states.append(np.expand_dims(s[3], axis=0))
            dones.append(s[4])

            # calculate weights
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)

        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        weights = torch.Tensor(np.reshape(weights, [self.batch_size, 1])).to(device)

        experiences = (states, actions, rewards, next_states, dones, weights, indices)

        return experiences

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.init_priority = max(self.init_priority, priority)


class PrioritizedReplayBufferfD(ReplayBuffer):
    """Create Prioritized Replay buffer with demo.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        buffer (deque): deque of experience replay buffer
        demo (deque): deque of demo replay buffer
        buffer_size (int): size of replay buffer for experience
        demo_size (int): size of replay buffer for demonstration
        total_size (int): sum of demo size and number of samples of experience
        alpha (float): alpha parameter for prioritized replay buffer
        epsilon_d (float) : epsilon_d parameter to update priority using demo
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        init_priority (float): lower bound of priority

        """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        demo: deque,
        alpha: float = 0.6,
        epsilon_d: float = 1.0,
    ):
        """Initialization.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            demo (deque): demonstration
            alpha (float): alpha parameter for prioritized replay buffer
            epsilon_d (float) : epsilon_d parameter to update priority using demo

        """
        super(PrioritizedReplayBufferfD, self).__init__(buffer_size, batch_size, demo)
        assert alpha >= 0
        self.buffer: deque = deque(maxlen=buffer_size)
        self.demo = demo
        self.buffer_size = buffer_size
        self.demo_size = len(demo)
        self.total_size = self.demo_size + len(self.buffer)
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_size + self.demo_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.init_priority = 1.0

        # for init priority of demo
        for _ in range(self.demo_size):
            self.sum_tree[self.tree_idx] = self.init_priority ** self.alpha
            self.min_tree[self.tree_idx] = self.init_priority ** self.alpha
            self.tree_idx += 1

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience and priority."""
        idx = self.tree_idx
        # buffer is full
        if (self.tree_idx + 1) % (self.buffer_size + self.demo_size) == 0:
            self.tree_idx = self.demo_size
        else:
            self.tree_idx = self.tree_idx + 1
        super().add(state, action, reward, next_state, done)

        self.sum_tree[idx] = self.init_priority ** self.alpha
        self.min_tree[idx] = self.init_priority ** self.alpha

        # update current total size
        self.total_size = self.demo_size + len(self.buffer)

    def extend(self, transitions: list):
        """Add experiences to memory."""
        raise NotImplementedError

    def _sample_proportional(self, batch_size: int) -> list:
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, self.total_size - 1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def sample(self, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        assert beta > 0

        indices = self._sample_proportional(self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        weights, eps_d = [], []

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.total_size) ** (-beta)

        for idx in indices:
            # sample from buffer
            if idx < self.demo_size:
                s = self.demo[idx]
                eps_d.append(self.epsilon_d)
            else:
                s = self.buffer[idx - self.demo_size]
                eps_d.append(0.0)

            # append transition info
            states.append(np.expand_dims(s[0], axis=0))
            actions.append(s[1])
            rewards.append(s[2])
            next_states.append(np.expand_dims(s[3], axis=0))
            dones.append(s[4])

            # calculate weights
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * self.total_size) ** (-beta)
            weights.append(weight / max_weight)

        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        weights = torch.Tensor(np.reshape(weights, [self.batch_size, 1])).to(device)

        experiences = (
            states,
            actions,
            rewards,
            next_states,
            dones,
            weights,
            indices,
            eps_d,
        )

        return experiences

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.total_size

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.init_priority = max(self.init_priority, priority)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.total_size
