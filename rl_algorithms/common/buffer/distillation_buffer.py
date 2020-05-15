# -*- coding: utf-8 -*-
"""Distillation buffer."""

from typing import Any, Tuple

import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DistillationBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
        obs_buf (np.ndarray): observations
        q_value_buf (np.ndarray): q_values for distillation
        buffer_size (int): size of buffers
        batch_size (int): batch size for training
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition

    """

    def __init__(
        self, buffer_size: int, batch_size: int,
    ):
        """Initialize a DistillationBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training

        """
        assert 0 < batch_size <= buffer_size

        self.obs_buf: np.ndarray = None
        self.q_value_buf: np.ndarray = None

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.length = 0
        self.idx = 0

    def add(self, transition: Tuple[np.ndarray, np.ndarray]) -> Tuple[Any, ...]:
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        state, q_values = transition
        if self.length == 0:
            self._initialize_buffers(state, q_values)

        self.obs_buf[self.idx] = state
        self.q_value_buf[self.idx] = q_values

        self.idx += 1
        self.idx = 0 if self.idx % self.buffer_size == 0 else self.idx
        self.length = min(self.length + 1, self.buffer_size)

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

    def _initialize_buffers(self, state: np.ndarray, q_values: np.ndarray) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray
        self.obs_buf = np.zeros(
            [self.buffer_size] + list(state.shape), dtype=state.dtype
        )
        self.q_value_buf = np.zeros(
            [self.buffer_size] + list(q_values.shape), dtype=float
        )

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length
