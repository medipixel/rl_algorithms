# -*- coding: utf-8 -*-
"""GaussianNoise class for baselines."""

import numpy as np


class GaussianNoise:
    """Gaussian Noise.

    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(
        self,
        seed: int,
        min_sigma: float = 1.0,
        max_sigma: float = 1.0,
        decay_period: int = 1000000,
    ):
        """Initialization."""
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        np.random.seed(seed)

    def sample(self, action_size: int, t: int = 0) -> float:
        """Get an action with gaussian noise."""
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.random.normal(size=action_size) * sigma
