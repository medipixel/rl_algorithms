# -*- coding: utf-8 -*-
"""Abstract class for computing reward.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

from abc import ABC, abstractmethod


class AbstractRewardFn(ABC):
    """Abstract class for computing reward.
       New compute_reward class should redefine __call__()

    Attributes:

    """

    @abstractmethod
    def __call__(self, *args):
        pass
