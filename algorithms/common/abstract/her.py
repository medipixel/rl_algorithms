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

    def __init__(self, reward_func: AbstractRewardFn):
        """Initialization.

        Args:
            reward_func (Callable): returns reward from state, action, next_state

        """
        self.reward_func = reward_func()

    @abstractmethod
    def get_desired_state(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def _set_final_state(self, transition: tuple) -> np.ndarray:
        pass

    def generate_transitions(
        self, transitions: list, desired_state: np.ndarray, is_demo: bool = False
    ) -> list:
        """Generate new transitions concatenated with desired states."""
        origin_transitions = list()
        new_transitions = list()
        final_state = self._set_final_state(transitions[-1])

        for transition in transitions:
            # process transitions with the initial goal state
            origin_transitions.append(self.__get_transition(transition, desired_state))
            if not is_demo:
                new_transitions.append(self.__get_transition(transition, final_state))

        return origin_transitions + new_transitions

    def __get_transition(self, transition: tuple, goal_state: np.ndarray):
        """Get a single transition concatenated with a goal state."""
        state, action, _, next_state, done = transition

        done = np.array_equal(next_state, goal_state)
        # TODO: should change no fix argument
        reward = self.reward_func(next_state, action, goal_state)
        state = np.concatenate((state, goal_state), axis=-1)
        next_state = np.concatenate((next_state, goal_state), axis=-1)

        return state, action, reward, next_state, done


class DemoHER(AbstractHER):
    """HER class using demonstration.

    Attributes:
        demo (list): demonstration
        demo_goal_indices (np.ndarray): indices about goal of demo list

    """

    def __init__(self, demo: list, reward_func: AbstractRewardFn):
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

    @abstractmethod
    def _set_demo_final_state(self, demo_transition: tuple):
        pass

    def generate_demo_transitions(self) -> list:
        """Return generated demo transitions for HER."""
        new_demo: list = list()

        # generate demo transitions
        prev_idx = 0
        for idx in self.demo_goal_indices:
            demo_final_state = self._set_demo_final_state(self.demo[idx])
            transitions = [self.demo[i] for i in range(prev_idx, idx + 1)]
            prev_idx = idx + 1

            transitions = self.generate_transitions(
                transitions, demo_final_state, is_demo=True
            )

            new_demo.extend(transitions)

        return new_demo

    @abstractmethod
    def get_desired_state(self, *args) -> np.ndarray:
        pass

    @abstractmethod
    def _set_final_state(self, transition: tuple) -> np.ndarray:
        pass
