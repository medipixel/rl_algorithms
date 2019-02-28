# -*- coding: utf-8 -*-
"""Hindsight Experience Replay.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1707.01495.pdf

"""

from typing import Callable

import numpy as np

from algorithms.common.helper_functions import fetch_desired_states_from_demo


def default_reward_func(
    state: np.ndarray, _: np.ndarray, goal_state: np.ndarray
) -> np.float64:
    """Default reward function."""
    eps = 1e-6
    if np.abs(state - goal_state).sum() < eps:
        return np.float64(0.0)
    else:
        return np.float64(-1.0)


class HER:
    """HER (final strategy).

    Attributes:
        desired_states (np.ndarray): desired states
        reward_func (Callable): returns reward from state, action, next_state

    """

    def __init__(self, demo_path: str, reward_func: Callable = default_reward_func):
        """Initialization.

        Args:
            demo_path (str): path of demonstration including desired states
            reward_func (Callable): returns reward from state, action, next_state
        """
        self.desired_states, self.demo_goal_indices = fetch_desired_states_from_demo(
            demo_path
        )
        self.reward_func = reward_func

    def sample_desired_state(self) -> np.ndarray:
        """Sample one of the desired states."""
        return np.random.choice(self.desired_states, 1).item()

    def generate_demo_transitions(self, demo: list) -> list:
        """Return generated demo transitions for HER."""
        new_demo: list = list()

        # generate demo transitions
        prev_idx = 0
        for idx in self.demo_goal_indices:
            demo_final_state = demo[idx][0]
            transitions = [demo[i] for i in range(prev_idx, idx + 1)]
            prev_idx = idx + 1

            transitions = self.generate_transitions(
                transitions, demo_final_state, demo=True
            )

            new_demo.extend(transitions)

        return new_demo

    def generate_transitions(
        self, transitions: list, desired_state: np.ndarray, demo: bool = False
    ) -> list:
        """Generate new transitions concatenated with desired states."""
        new_transitions = list()
        final_state = transitions[-1][0]

        for transition in transitions:
            # process transitions with the initial goal state
            new_transitions.append(self.__get_transition(transition, desired_state))
            if not demo:
                new_transitions.append(self.__get_transition(transition, final_state))

        return new_transitions

    def __get_transition(self, transition: tuple, goal_state: np.ndarray):
        """Get a single transition concatenated with a goal state."""
        state, action, _, next_state, done = transition

        done = np.array_equal(state, goal_state)
        reward = self.reward_func(state, action, goal_state)
        state = np.concatenate((state, goal_state), axis=-1)
        next_state = np.concatenate((next_state, goal_state), axis=-1)

        return state, action, reward, next_state, done
