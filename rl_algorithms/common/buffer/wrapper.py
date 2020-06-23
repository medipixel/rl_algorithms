# -*- coding: utf-8 -*-
"""Wrappers for buffer.

- Author: Kyunghwan Kim, Euijin Jeong
- Contacts: kh.kim@medipixel.io
            euijin.jeong@medipixel.io
- Paper: https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import random
from typing import Any, Tuple

import numpy as np
import torch

from rl_algorithms.common.abstract.buffer import BaseBuffer, BufferWrapper
from rl_algorithms.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedBufferWrapper(BufferWrapper):
    """Prioritized Experience Replay wrapper for Buffer.


    Refer to OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        buffer (Buffer): Hold replay buffer as an attribute
        alpha (float): alpha parameter for prioritized replay buffer
        epsilon_d (float): small positive constants to add to the priorities
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        _max_priority (float): max priority
    """

    def __init__(
        self, base_buffer: BaseBuffer, alpha: float = 0.6, epsilon_d: float = 1.0
    ):
        """Initialize.

        Args:
            base_buffer (Buffer): ReplayBuffer which should be hold
            alpha (float): alpha parameter for prioritized replay buffer
            epsilon_d (float): small positive constants to add to the priorities

        """
        BufferWrapper.__init__(self, base_buffer)
        assert alpha >= 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer.max_len:
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
            if self.tree_idx % self.buffer.max_len == 0:
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
        experiences = self.buffer.sample(indices)

        return experiences + (weights, indices, eps_d)

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)
