# -*- coding: utf-8 -*-
"""Environment wrapper class for Pong.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Tuple, Union

import cv2
import gym
import numpy as np

TERMINAL_SCORE = 11


class EarlyTerminationWrapper(gym.Wrapper):
    """Terminate the game if one reachs the terminal score."""

    def __init__(self, env: gym.Env):
        """Initialization."""
        super(EarlyTerminationWrapper, self).__init__(env)
        self.enemy_score = 0
        self.my_score = 0
        self.terminal_score = TERMINAL_SCORE

    # pylint: disable=method-hidden
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Counter each score."""
        obs, reward, done, info = self.env.step(action)
        if reward < 0:
            self.enemy_score += 1
        if reward > 0:
            self.my_score += 1

        if (self.enemy_score == self.terminal_score) or (
            self.my_score == self.terminal_score
        ):
            done = True
            self.enemy_score = 0
            self.my_score = 0

        return obs, reward, done, info

    # pylint: disable=method-hidden
    def reset(self) -> np.ndarray:
        """Do nothing."""
        return self.env.reset()


class ObsPreprocessingWrapper(gym.ObservationWrapper):
    """Return preprocessed observations."""

    def __init__(self, env: gym.Env):
        """Initialization."""
        super(ObsPreprocessingWrapper, self).__init__(env)
        self.current_phi = np.zeros(1)

    def observation(self, obs) -> np.ndarray:
        """Preprocess Observation."""
        obs = obs.astype("float32")
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84)) / 127.5 - 1.0

        if self.current_phi.size == 1:  # at the beginning
            self.current_phi = np.stack([obs, obs, obs, obs])
        else:
            self.current_phi = self._phi(obs)

        return self.current_phi

    def _phi(self, obs: np.ndarray) -> np.ndarray:
        """Generate 4-channel state."""
        new_phi = np.zeros((4, 84, 84), dtype=np.float32)
        new_phi[:3] = self.current_phi[1:]
        new_phi[-1] = obs

        return new_phi


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


WRAPPERS = [
    # EarlyTerminationWrapper,
    ObsPreprocessingWrapper,
    ClippedRewardsWrapper,
    ReducedActionWrapper,
]
