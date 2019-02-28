# -*- coding: utf-8 -*-
"""Environment wrapper class for Pong-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Union

import cv2
import gym
import numpy as np


class ObservationPreprocessor(gym.ObservationWrapper):
    """Return preprocessed observations."""

    def __init__(self, env: gym.Env):
        """Initialization."""
        super(ObservationPreprocessor, self).__init__(env)
        self.current_phi = np.zeros(1)
        self.is_started = False

    def observation(self, obs) -> np.ndarray:
        """Preprocess Observation."""
        obs = self.rgb2gray(obs)

        if not self.is_started:  # at the beginning
            self.current_phi = np.stack([obs, obs, obs, obs])
            self.is_started = True
        else:
            self.current_phi = self._phi(obs)

        return self.current_phi

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Generate 4-channel state."""
        new_phi = np.zeros((4, 84, 84), dtype=np.float32)
        new_phi[:3] = self.current_phi[1:]
        new_phi[-1] = x

        return new_phi

    @staticmethod
    def rgb2gray(x: np.ndarray) -> np.ndarray:
        """Convert rgb image to gray."""
        x = x.astype("float32")
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(x, (84, 84)) / 127.5 - 1.0

        return x


class ClippedRewardsWrapper(gym.RewardWrapper):
    """Change all positive to 1, negative to -1 and keep zero."""

    def reward(self, reward):
        return self.signed_reward(reward)

    @staticmethod
    def signed_reward(reward):
        return np.sign(reward)


class ReducedActionWrapper(gym.ActionWrapper):
    """Use only available 3 actions."""

    def __init__(self, env: gym.Env):
        """Initialization."""
        self.valid_actions = np.array([0, 2, 5])
        self.action_dim = len(self.valid_actions)
        env.action_space.n = self.action_dim
        env.action_space.sample = self.sample_actions

        super(ReducedActionWrapper, self).__init__(env)

    def action(self, idx: Union[np.int64, np.ndarray]) -> np.ndarray:
        """Return to available actions"""
        return self.valid_actions[idx].squeeze()

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_actions(self) -> int:
        """Get random discrete action."""
        return np.random.choice(self.action_dim)


WRAPPERS = [ObservationPreprocessor, ClippedRewardsWrapper, ReducedActionWrapper]
