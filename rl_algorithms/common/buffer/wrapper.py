# -*- coding: utf-8 -*-
"""Prioritized Replay buffer for algorithms.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import random
from typing import Any, Tuple

import numpy as np
import torch

from rl_algorithms.common.abstract.buffer import BufferWrapper
from rl_algorithms.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PERWrapper(BufferWrapper):
    """Fixed-size buffer to store experience tuples.

    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rews_buf (np.ndarray): rewards
        next_obs_buf (np.ndarray): next observations
        done_buf (np.ndarray): dones
        n_step_buffer (deque): recent n transitions
        n_step (int): step size for n-step transition
        gamma (float): discount factor
        buffer_size (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(self, replay_buffer, alpha: float = 0.6, epsilon_d: float = 1.0):
        """Initialize.

        Args:
            batch_size (int): size of a batched sampled from replay buffer for training
            alpha (float): alpha parameter for prioritized replay buffer

        """
        super().__init__(replay_buffer)
        assert alpha >= 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

        # for init priority of demo
        self.tree_idx = self.buffer.demo_size
        for i in range(self.buffer.demo_size):
            self.sum_tree[i] = self._max_priority ** self.alpha
            self.min_tree[i] = self._max_priority ** self.alpha

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> Tuple[Any, ...]:
        """Add experience and priority."""
        n_step_transition = self.buffer.add(transition)
        if n_step_transition:
            self.sum_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.min_tree[self.tree_idx] = self._max_priority ** self.alpha

            self.tree_idx += 1
            if self.tree_idx % self.buffer.buffer_size == 0:
                self.tree_idx = self.buffer.demo_size

        return n_step_transition

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
        assert len(self.buffer) >= self.buffer.batch_size
        assert beta > 0

        indices = self._sample_proportional(self.buffer.batch_size)

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        # calculate weights
        weights_, eps_d = [], []
        for i in indices:
            eps_d.append(self.epsilon_d if i < self.buffer.demo_size else 0.0)
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights_.append(weight / max_weight)

        weights = np.array(weights_)
        eps_d = np.array(eps_d)

        weights = weights.reshape(-1, 1)

        states, actions, rewards, next_states, dones = self.buffer.sample(indices)

        return states, actions, rewards, next_states, dones, weights, indices, eps_d

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)
