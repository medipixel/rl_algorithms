# -*- coding: utf-8 -*-
"""Abstract class for computing reward.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
from abc import ABC, abstractmethod

import numpy as np


class RewardFn(ABC):
    """Abstract class for computing reward.
    New compute_reward class should redefine __call__()

    """

    @abstractmethod
    def __call__(self, transition: tuple, goal_state: np.ndarray) -> np.float64:
        pass
