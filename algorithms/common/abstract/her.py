# -*- coding: utf-8 -*-
"""Abstract class used for Hindsight Experience Replay.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1707.01495.pdf
"""

from abc import ABC, abstractmethod

import numpy as np

from algorithms.common.abstract.reward_fn import AbstractRewardFn


class AbstractHER(ABC):
    """Abstract class for HER (final strategy).

    Attributes:
        reward_func (Callable): returns reward from state, action, next_state

    """

    def __init__(self, *args, reward_func: AbstractRewardFn):
        """Initialization.

        Args:
            reward_func (Callable): returns reward from state, action, next_state

        """
        self.args = args
        self.reward_func = reward_func()

    @abstractmethod
    def get_desired_state(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def _get_final_state(self, transition: tuple) -> np.ndarray:
        pass

    @abstractmethod
    def generate_demo_transitions(self, demo: list) -> list:
        pass

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
            origin_transitions.append(self.__get_transition(transition, desired_state))
            if not is_demo and score <= success_score:
                new_transitions.append(self.__get_transition(transition, final_state))

        return origin_transitions + new_transitions

    def __get_transition(self, transition: tuple, goal_state: np.ndarray):
        """Get a single transition concatenated with a goal state."""
        state, action, _, next_state, done = transition

        done = np.array_equal(next_state, goal_state)
        reward = self.reward_func(next_state, action, goal_state)
        state = np.concatenate((state, goal_state), axis=-1)
        next_state = np.concatenate((next_state, goal_state), axis=-1)

        return state, action, reward, next_state, done
