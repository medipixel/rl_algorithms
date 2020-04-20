# -*- coding: utf-8 -*-
"""HER class and reward function for Behavior Cloning.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
"""
from typing import Callable, Tuple

import numpy as np

from rl_algorithms.common.abstract.her import HER
from rl_algorithms.common.abstract.reward_fn import RewardFn
from rl_algorithms.registry import HERS


class L1DistanceRewardFn(RewardFn):
    def __call__(self, transition: tuple, goal_state: np.ndarray) -> np.float64:
        """L1 Distance reward function."""
        next_state = transition[3]
        eps = 1e-6
        if np.abs(next_state - goal_state).sum() < eps:
            return np.float64(0.0)
        else:
            return np.float64(-1.0)


l1_distance_reward_fn = L1DistanceRewardFn()


@HERS.register_module
class LunarLanderContinuousHER(HER):
    """HER for LunarLanderContinuous-v2 environment.

    Attributes:
        demo_goal_indices (np.ndarray): indices about goal of demo list
        desired_states (np.ndarray): desired states from demonstration

    """

    def __init__(
        self,
        reward_fn: Callable[[tuple, np.ndarray], np.float64] = l1_distance_reward_fn,
    ):
        """Initialize."""
        HER.__init__(self, reward_fn=reward_fn)
        self.is_goal_in_state = False

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


class ReacherRewardFn(RewardFn):
    def __call__(self, transition: tuple, _) -> np.float64:
        """Reward function for Reacher-v2 environment."""
        state, action = transition[0:2]
        diff_vec = state[-3:]
        reward_dist = -1 * np.linalg.norm(diff_vec)
        reward_ctrl = -np.square(action).sum()

        return reward_dist + reward_ctrl


reacher_reward_fn = ReacherRewardFn()


@HERS.register_module
class ReacherHER(HER):
    """HER for Reacher-v2 environment."""

    def __init__(
        self, reward_fn: Callable[[tuple, np.ndarray], np.float64] = reacher_reward_fn
    ):
        """Initialize."""
        HER.__init__(self, reward_fn=reward_fn)
        self.is_goal_in_state = True

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

    def _get_transition(
        self, transition: tuple, goal_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.float64, np.ndarray, bool]:
        """Get a single transition concatenated with a goal state."""
        state, action, _, next_state, done = transition

        reward = self.reward_fn(transition, goal_state)
        state_ = state
        state_[4:6] = goal_state
        next_state_ = next_state
        next_state_[4:6] = goal_state

        return state_, action, reward, next_state_, done
