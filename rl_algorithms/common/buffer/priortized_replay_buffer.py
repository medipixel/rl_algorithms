# -*- coding: utf-8 -*-
"""Prioritized Replay buffer for algorithms.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import random
from typing import Any, List, Tuple

import numpy as np
import torch

from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer(ReplayBuffer):
    """Create Prioritized Replay buffer.

    Refer to OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        alpha (float): alpha parameter for prioritized replay buffer
        epsilon_d (float): small positive constants to add to the priorities
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        _max_priority (float): max priority
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int = 32,
        gamma: float = 0.99,
        n_step: int = 1,
        alpha: float = 0.6,
        epsilon_d: float = 1.0,
        demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
    ):
        """Initialize.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            alpha (float): alpha parameter for prioritized replay buffer

        """
        super(PrioritizedReplayBuffer, self).__init__(
            buffer_size, batch_size, gamma, n_step, demo
        )
        assert alpha >= 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

        # for init priority of demo
        self.tree_idx = self.demo_size
        for i in range(self.demo_size):
            self.sum_tree[i] = self._max_priority ** self.alpha
            self.min_tree[i] = self._max_priority ** self.alpha

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> Tuple[Any, ...]:
        """Add experience and priority."""
        n_step_transition = super().add(transition)
        if n_step_transition:
            self.sum_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.min_tree[self.tree_idx] = self._max_priority ** self.alpha

            self.tree_idx += 1
            if self.tree_idx % self.buffer_size == 0:
                self.tree_idx = self.demo_size

        return n_step_transition

    def _sample_proportional(self, batch_size: int) -> list:
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def sample(self, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:  # type: ignore
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional(self.batch_size)

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        weights_, eps_d = [], []
        for i in indices:
            eps_d.append(self.epsilon_d if i < self.demo_size else 0.0)
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights_.append(weight / max_weight)

        weights = np.array(weights_)
        eps_d = np.array(eps_d)

        weights = torch.FloatTensor(weights.reshape(-1, 1)).to(device)

        if torch.cuda.is_available():
            weights = weights.cuda(non_blocking=True)

        states, actions, rewards, next_states, dones = super().sample(indices)

        return states, actions, rewards, next_states, dones, weights, indices, eps_d

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)


class DistillationPER(PrioritizedReplayBuffer):
    """PER with policy distillation."""

    # pylint: disable=attribute-defined-outside-init
    def add(
        self,
        transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool],
        q_values: np.ndarray,
    ) -> Tuple[Any, ...]:
        """Add experience and priority."""
        if len(self.n_step_buffer) < self.n_step:
            self.q_value_buf = np.zeros(
                [self.buffer_size] + list(q_values.shape), dtype=float
            )

        n_step_transition = super().add(transition)

        self.q_value_buf[self.idx] = q_values

        return n_step_transition

    def sample_for_diltillation(self):
        """Sample a batch of state and Q-value for policy distillation."""
        assert len(self) >= self.batch_size

        indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states = torch.FloatTensor(self.obs_buf[indices]).to(device)
        q_values = torch.FloatTensor(self.q_value_buf[indices]).to(device)

        if torch.cuda.is_available():
            states = states.cuda(non_blocking=True)
            q_values = q_values.cuda(non_blocking=True)

        return states, q_values
