# -*- coding: utf-8 -*-
"""Hindsight Experience Replay.

- Author: Curt Park, Kh Kim
- Contact: curt.park@medipixel.io
           kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1707.01495.pdf

"""

from typing import Callable

import numpy as np

from algorithms.common.abstract.her import AbstractHER


def default_reward_func(
    state: np.ndarray, _: np.ndarray, goal_state: np.ndarray
) -> np.float64:
    """Default reward function."""
    eps = 1e-6
    if np.abs(state - goal_state).sum() < eps:
        return np.float64(0.0)
    else:
        return np.float64(-1.0)


class DemoHER(AbstractHER):
    """HER class using demonstration.

    Attributes:
        demo (list): demonstration
        demo_goal_indices (np.ndarray): indices about goal of demo list

    """

    def __init__(self, demo: list, reward_func: Callable):
        """Initialization.

        Args:
            demo (list): demonstration
            reward_func (Callable): returns reward from state, action, next_state

        """
        AbstractHER.__init__(self, reward_func)

        self.demo = demo
        self.demo_goal_indices = self.fetch_goal_indices_from_demo()

    def fetch_goal_indices_from_demo(self) -> np.ndarray:
        """Return goal indices from demonstration data."""
        demo = np.array(self.demo)
        goal_indices = np.where(demo[:, 4])[0]

        return goal_indices

    def generate_demo_transitions(self) -> list:
        """Return generated demo transitions for HER."""
        new_demo: list = list()

        # generate demo transitions
        prev_idx = 0
        for idx in self.demo_goal_indices:
            demo_final_state = self.demo[idx][0]
            transitions = [self.demo[i] for i in range(prev_idx, idx + 1)]
            prev_idx = idx + 1

            transitions = self.generate_transitions(
                transitions, demo_final_state, is_demo=True
            )

            new_demo.extend(transitions)

        return new_demo

    def set_desired_states(self):
        pass

    def get_desired_state(self) -> np.ndarray:
        pass


class LunarLanderContinuousHER(DemoHER):
    """HER for LunarLanderContinuous-v2 environment.

    Attributes:
        desired_states (np.ndarray): desired states from demonstration

    """

    def __init__(self, demo: list, reward_func: Callable = default_reward_func):
        """Initialization."""
        DemoHER.__init__(self, demo, reward_func)
        self.desired_states = self.fetch_desired_states_from_demo()

    def fetch_desired_states_from_demo(self) -> np.ndarray:
        """Return desired goal states from demonstration data."""
        demo = np.array(self.demo)
        goal_states = demo[self.demo_goal_indices][:, 0]

        return goal_states

    def get_desired_state(self) -> np.ndarray:
        """Sample one of the desired states."""
        return np.random.choice(self.desired_states, 1).item()


class ReacherHER(DemoHER):
    """HER for Reacher-v2 environment."""

    def __init__(self, demo: list, reward_func: Callable):
        """Initialization."""
        DemoHER.__init__(self, demo, reward_func)
