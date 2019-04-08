# -*- coding: utf-8 -*-
"""Utils for examples on LunarLanderContinuous-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import numpy as np

from algorithms.common.abstract.her import HER as AbstractHER
from algorithms.common.abstract.reward_fn import RewardFn


class L1DistanceRewardFn(RewardFn):
    def __call__(self, transition: tuple, goal_state: np.ndarray) -> np.float64:
        """L1 Distance reward function."""
        next_state = transition[3]
        eps = 1e-6
        if np.abs(next_state - goal_state).sum() < eps:
            return np.float64(0.0)
        else:
            return np.float64(-1.0)


class LunarLanderContinuousHER(AbstractHER):
    """HER for LunarLanderContinuous-v2 environment.

    Attributes:
        demo_goal_indices (np.ndarray): indices about goal of demo list
        desired_states (np.ndarray): desired states from demonstration

    """

    def __init__(self, reward_func: RewardFn = L1DistanceRewardFn):
        """Initialization."""
        AbstractHER.__init__(self, reward_func=reward_func)

    # pylint: disable=attribute-defined-outside-init
    def fetch_desired_states_from_demo(self, demo: list):
        """Return desired goal states from demonstration data."""
        np_demo: np.ndarray = np.array(demo)
        self.demo_goal_indices: np.ndarray = np.where(np_demo[:, 4])[0]
        self.desired_states: np.ndarray = np_demo[self.demo_goal_indices][:, 0]

    def get_desired_state(self, *args) -> np.ndarray:
        """Sample one of the desired states."""
        return np.random.choice(self.desired_states, 1).item()

    def _get_final_state(self, transition: tuple) -> np.ndarray:
        """Get final state from transitions for making HER transitions."""
        return transition[0]

    def generate_demo_transitions(self, demo: list) -> list:
        """Return generated demo transitions for HER."""
        new_demo: list = list()

        # generate demo transitions
        prev_idx = 0
        for idx in self.demo_goal_indices:
            demo_final_state = self._get_final_state(demo[idx])
            transitions = [demo[i] for i in range(prev_idx, idx + 1)]
            prev_idx = idx + 1

            transitions = self.generate_transitions(
                transitions, demo_final_state, 0, is_demo=True
            )

            new_demo.extend(transitions)

        return new_demo
