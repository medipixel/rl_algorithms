# -*- coding: utf-8 -*-
"""Utils for examples on Reacher-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import numpy as np

from algorithms.common.abstract.her import AbstractHER
from algorithms.common.abstract.reward_fn import AbstractRewardFn


class ReacherRewardFn(AbstractRewardFn):
    def __call__(self, state: np.ndarray, action: np.ndarray, _) -> np.float64:
        """Reward function for Reacher-v2 environment."""
        diff_vec = state[-3:]
        reward_dist = -1 * np.linalg.norm(diff_vec)
        reward_ctrl = -np.square(action).sum()

        return reward_dist + reward_ctrl


class ReacherHER(AbstractHER):
    """HER for Reacher-v2 environment."""

    def __init__(self, *args, reward_func: AbstractRewardFn = ReacherRewardFn):
        """Initialization."""
        AbstractHER.__init__(self, *args, reward_func=reward_func)

    def get_desired_state(self, *args) -> np.ndarray:
        return np.array([])

    def _get_final_state(self, transition: tuple) -> np.ndarray:
        return transition[0][8:10] + transition[0][2:4]

    def generate_demo_transitions(self, demo: list) -> list:
        return demo

    def generate_transitions(
        self,
        transitions: list,
        desired_state: np.ndarray,
        success_score: float,
        is_demo: bool = False,
    ) -> list:
        """Generate new transitions concatenated with desired states."""
        origin_transitions = list()
        new_transitions = list()
        final_state = self._get_final_state(transitions[-1])
        score = np.sum(np.array(transitions), axis=0)[2]

        for transition in transitions:
            # process transitions with the initial goal state
            origin_transitions.append(transition)
            if not is_demo and score <= success_score:
                new_transitions.append(self.__get_transition(transition, final_state))

        return origin_transitions + new_transitions

    def __get_transition(self, transition: tuple, goal_state: np.ndarray):
        """Get a single transition concatenated with a goal state."""
        state, action, _, next_state, done = transition

        reward = self.reward_func(next_state, action, goal_state)
        state_ = state
        state_[4:6] = goal_state
        next_state_ = next_state
        next_state_[4:6] = goal_state

        return state_, action, reward, next_state_, done
