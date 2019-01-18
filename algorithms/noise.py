# -*- coding: utf-8 -*-
"""Noise classes for baselines."""

import copy
import random
import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) +\
            self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class GaussianNoise(object):
    """Gaussian Noise.

    Taken from https://github.com/vitchyr/rlkit
    """

    def __init__(self, action_low, action_high, min_sigma=1.0,
                 max_sigma=1.0, decay_period=1000000):
        """Initialization."""
        self.low = action_low
        self.high = action_high
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def sample(self, action_size, t=0):
        """Get an action with gaussian noise."""
        sigma = \
            self.max_sigma - (self.max_sigma - self.min_sigma) * \
            min(1.0, t / self.decay_period)
        return np.random.normal(size=action_size) * sigma
