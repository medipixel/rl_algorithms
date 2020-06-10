# -*- coding: utf-8 -*-
"""Abstract Buffer & BufferWrapper class.

- Author: Euijin Jeong
- Contact: euijin.jeong@medipixel.io
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class BaseBuffer(ABC):
    """Abstract Buffer used for replay buffer."""

    @abstractmethod
    def add(self, transition: Tuple[Any, ...]) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def sample(self) -> Tuple[np.ndarray, ...]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class BufferWrapper(BaseBuffer):
    """Abstract BufferWrapper used for buffer wrapper.

    Attributes:
        buffer (Buffer): Hold replay buffer as am attribute
    """

    def __init__(self, base_buffer: BaseBuffer):
        """Initialize a ReplayBuffer object.

        Args:
            base_buffer (int): ReplayBuffer which should be hold
        """
        self.buffer = base_buffer

    def add(self, transition: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return self.buffer.add(transition)

    def sample(self) -> Tuple[np.ndarray, ...]:
        return self.buffer.sample()

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)
