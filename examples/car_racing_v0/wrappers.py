# -*- coding: utf-8 -*-
"""Converter - continuous actions to discrete for CarRacing-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import itertools as it
from typing import Union

import gym
import numpy as np

import examples.car_racing_v0.utils as env_utils


class Continuous2Discrete(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def __init__(self, env: gym.Env):
        """Initialization."""
        self.all_actions = np.array(
            [k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])]
        )
        self.action_dim = len(self.all_actions)
        self.gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in self.all_actions])
        env.action_space.sample = self.sample_actions

        super(Continuous2Discrete, self).__init__(env)

    def action(self, idx: Union[np.int64, np.ndarray]) -> np.ndarray:
        """Change discrete actions to continuous."""
        return self.all_actions[idx].squeeze()

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change continuous actions to discrete."""
        raise NotImplementedError

    def sample_actions(self) -> int:
        """Get random discrete action."""
        action_weights = 14.0 * self.gas_actions + 1.0
        action_weights /= np.sum(action_weights)
        return np.random.choice(self.action_dim, p=action_weights)


class PreprocessedObservation(gym.ObservationWrapper):
    """Return preprocessed observations."""

    def observation(self, observation) -> np.ndarray:
        """Preprocess Observation"""
        return env_utils.process_image(observation)
