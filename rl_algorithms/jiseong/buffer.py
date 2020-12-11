# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
import torch

from rl_algorithms.common.abstract.buffer import BaseBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class JBuffer(BaseBuffer):
    """Fixed-size buffer to store experience tuples.

    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rews_buf (np.ndarray): rewards
        next_obs_buf (np.ndarray): next observations
        done_buf (np.ndarray): dones
        n_step_buffer (deque): recent n transitions
        n_step (int): step size for n-step transition
        gamma (float): discount factor
        max_len (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(
        self, max_len: int, batch_size: int, gamma: float = 0.99, n_step: int = 1,
    ):
        """Initialize a ReplayBuffer object.

        Args:
            max_len (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human play
        """
        assert 0 < batch_size <= max_len
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= max_len

        self.obs_buf: np.ndarray = None
        self.state_info_buf: np.ndarray = None
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.next_obs_buf: np.ndarray = None
        self.next_state_info_buf: np.ndarray = None
        self.done_buf: np.ndarray = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.max_len = max_len
        self.batch_size = batch_size
        self.length = 0
        self.idx = 0

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> Tuple[Any, ...]:
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        assert len(transition) == 7, "Inappropriate transition size"
        assert isinstance(transition[0], np.ndarray)
        assert isinstance(transition[1], np.ndarray)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, state_info, action = transition[:3]
            self._initialize_buffers(state, state_info, action)

        (
            curr_state,
            state_info,
            action,
            reward,
            next_state,
            next_state_info,
            done,
        ) = transition
        self.obs_buf[self.idx] = curr_state
        self.state_info_buf[self.idx] = state_info
        self.acts_buf[self.idx] = action
        self.rews_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
        self.next_state_info_buf[self.idx] = next_state_info
        self.done_buf[self.idx] = done

        self.idx += 1
        self.idx = self.demo_size if self.idx % self.max_len == 0 else self.idx
        self.length = min(self.length + 1, self.max_len)

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def extend(
        self, transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
    ):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(transition)

    def sample(self, indices: List[int] = None) -> Tuple[np.ndarray, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states = self.obs_buf[indices]
        state_info = self.state_info_buf[indices]
        actions = self.acts_buf[indices]
        rewards = self.rews_buf[indices].reshape(-1, 1)
        next_states = self.next_obs_buf[indices]
        next_state_info = self.next_state_info_buf[indices]
        dones = self.done_buf[indices].reshape(-1, 1)

        return states, state_info, actions, rewards, next_states, next_state_info, dones

    def _initialize_buffers(
        self, state: np.ndarray, state_info, action: np.ndarray
    ) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray
        self.obs_buf = np.zeros([self.max_len] + list(state.shape), dtype=state.dtype)
        self.state_info_buf = np.zeros(
            [self.max_len] + list(state_info.shape), dtype=state_info.dtype
        )
        self.acts_buf = np.zeros(
            [self.max_len] + list(action.shape), dtype=action.dtype
        )
        self.rews_buf = np.zeros([self.max_len], dtype=float)
        self.next_obs_buf = np.zeros(
            [self.max_len] + list(state.shape), dtype=state.dtype
        )
        self.next_state_info_buf = np.zeros_like(self.state_info_buf)
        self.done_buf = np.zeros([self.max_len], dtype=float)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length
