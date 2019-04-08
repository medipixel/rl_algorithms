# -*- coding: utf-8 -*-
"""Utils for examples on Reacher-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import numpy as np

from algorithms.common.abstract.her import HER as AbstractHER
from algorithms.common.abstract.reward_fn import RewardFn


class ReacherRewardFn(RewardFn):
    def __call__(self, transition: tuple, _) -> np.float64:
        """Reward function for Reacher-v2 environment."""
        state, action = transition[0:2]
        diff_vec = state[-3:]
        reward_dist = -1 * np.linalg.norm(diff_vec)
        reward_ctrl = -np.square(action).sum()

        return reward_dist + reward_ctrl


class ReacherHER(AbstractHER):
    """HER for Reacher-v2 environment."""

    def __init__(self, reward_func: RewardFn = ReacherRewardFn):
        """Initialization."""
        AbstractHER.__init__(self, reward_func=reward_func)

    def fetch_desired_states_from_demo(self, _: list):
        """Return desired goal states from demonstration data.

        DO NOT use this method because demo states have a goal position.
        """
        raise Exception("Do not use this method.")

    def get_desired_state(self, *args) -> np.ndarray:
        """Sample one of the desired states.

        Returns an empty array since demo states have a goal position.
        """
        return np.array([])

    def _get_final_state(self, transition_final: tuple) -> np.ndarray:
        """Get a finger-tip position from the final transition."""
        return transition_final[0][8:10] + transition_final[0][2:4]

    def generate_demo_transitions(self, demo: list) -> list:
        """Return generated demo transitions for HER.

        Works as an identity function in this class.
        """
        return demo

    def _append_origin_transitions(
        self, origin_transitions: list, transition: tuple, _: np.ndarray
    ):
        """Append original transitions for training."""
        origin_transitions.append(transition)

    def _get_transition(self, transition: tuple, goal_state: np.ndarray) -> tuple:
        """Get a single transition concatenated with a goal state."""
        state, action, _, next_state, done = transition

        reward = self.reward_func(transition, goal_state)
        state_ = state
        state_[4:6] = goal_state
        next_state_ = next_state
        next_state_[4:6] = goal_state

        return state_, action, reward, next_state_, done
