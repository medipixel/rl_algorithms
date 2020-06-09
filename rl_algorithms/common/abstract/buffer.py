from typing import Any, Tuple

import numpy as np


class Buffer:
    def add(self, transition: Tuple[Any, ...]) -> Tuple[Any, ...]:
        pass

    def sample(self) -> Tuple[np.ndarray, ...]:
        pass

    def __len__(self) -> int:
        pass


class BufferWrapper(Buffer):
    def __init__(self, replay_buffer: Buffer):
        self.buffer = replay_buffer

    def add(self, transition: Tuple[Any, ...]) -> Tuple[Any, ...]:
        return self.buffer.add(transition)

    def sample(self) -> Tuple[np.ndarray, ...]:
        return self.buffer.sample()

    def __len__(self) -> int:
        return len(self.buffer)
