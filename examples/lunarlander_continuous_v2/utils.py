# -*- coding: utf-8 -*-
"""Utils for examples on LunarLanderContinuous-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import numpy as np

from algorithms.common.abstract.her import DemoHER
from algorithms.common.abstract.reward_fn import AbstractRewardFn


class L1DistanceRewardFn(AbstractRewardFn):
    def __call__(self, next_state: np.ndarray, _, goal_state: np.ndarray) -> np.float64:
        """L1 Distance reward function."""
        eps = 1e-6
        if np.abs(next_state - goal_state).sum() < eps:
            return np.float64(0.0)
        else:
            return np.float64(-1.0)


class LunarLanderContinuousHER(DemoHER):
    """HER for LunarLanderContinuous-v2 environment.

    Attributes:
        desired_states (np.ndarray): desired states from demonstration

    """

    def __init__(self, demo: list, reward_func: AbstractRewardFn = L1DistanceRewardFn):
        """Initialization."""
        DemoHER.__init__(self, demo, reward_func)
        self.desired_states = self.fetch_desired_states_from_demo()

    def fetch_desired_states_from_demo(self) -> np.ndarray:
        """Return desired goal states from demonstration data."""
        demo = np.array(self.demo)
        goal_states = demo[self.demo_goal_indices][:, 0]

        return goal_states

    def get_desired_state(self, *args) -> np.ndarray:
        """Sample one of the desired states."""
        return np.random.choice(self.desired_states, 1).item()

    def _set_final_state(self, transition: tuple) -> np.ndarray:
        return transition[0]

    def _set_demo_final_state(self, demo_transition: tuple):
        return demo_transition[0]
