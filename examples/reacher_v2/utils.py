# -*- coding: utf-8 -*-
"""Utils for examples on Reacher-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import numpy as np

from algorithms.common.abstract.her import DemoHER
from algorithms.common.abstract.reward_fn import AbstractRewardFn


class ReacherRewardFn(AbstractRewardFn):
    def __call__(self, state: np.ndarray, action: np.ndarray, _) -> np.float64:
        """Reward function for Reacher-v2 environment."""
        diff_vec = state[-3:]
        reward_dist = -1 * np.linalg.norm(diff_vec)
        reward_ctrl = -np.square(action).sum()

        return reward_dist + reward_ctrl


class ReacherHER(DemoHER):
    """HER for Reacher-v2 environment."""

    def __init__(self, demo: list, reward_func: AbstractRewardFn = ReacherRewardFn):
        """Initialization."""
        DemoHER.__init__(self, demo, reward_func)

    def get_desired_state(self, env) -> np.ndarray:
        """Sample one of the desired states."""
        return env.unwrapped.goal

    def _set_final_state(self, transition: tuple) -> np.ndarray:
        return transition[0][2:4]

    def _set_demo_final_state(self, demo_transition: tuple) -> np.ndarray:
        return demo_transition[0][2:4]
