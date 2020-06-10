# -*- coding: utf-8 -*-
"""Abstract Buffer & BufferWrapper class.
- Author: Euijin Jeong
- Contact: euijin.jeong@medipixel.io
"""

from typing import Any, Tuple

import numpy as np


class Buffer:
    """Abstract Buffer used for replay buffer."""

    def add(self, transition: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Add a new experience to memory."""

    def sample(self) -> Tuple[np.ndarray, ...]:
        """Sample a batch of experiences from memory."""

    def __len__(self) -> int:
        """Return the current size of internal memory."""


class BufferWrapper(Buffer):
    """Abstract BufferWrapper used for buffer wrapper.
    Attributes:
        buffer (Buffer): Hold replay buffer as am attribute.
    """

    def __init__(self, replay_buffer: Buffer):
        self.buffer = replay_buffer

    def add(self, transition: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return self.buffer.add(transition)

    def sample(self) -> Tuple[np.ndarray, ...]:
        return self.buffer.sample()

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)
