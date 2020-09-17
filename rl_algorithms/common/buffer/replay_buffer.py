# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
import torch

from rl_algorithms.common.abstract.buffer import BaseBuffer
from rl_algorithms.common.helper_functions import get_n_step_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(BaseBuffer):
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
        self,
        max_len: int,
        batch_size: int,
        gamma: float = 0.99,
        n_step: int = 1,
        demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
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
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.next_obs_buf: np.ndarray = None
        self.done_buf: np.ndarray = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.max_len = max_len
        self.batch_size = batch_size
        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size

        # demo may have empty tuple list [()]
        if self.demo and self.demo[0]:
            self.max_len += self.demo_size
            self.length += self.demo_size
            for idx, d in enumerate(self.demo):
                state, action, reward, next_state, done = d
                if idx == 0:
                    action = (
                        np.array(action).astype(np.int64)
                        if isinstance(action, int)
                        else action
                    )
                    self._initialize_buffers(state, action)
                self.obs_buf[idx] = state
                self.acts_buf[idx] = np.array(action)
                self.rews_buf[idx] = reward
                self.next_obs_buf[idx] = next_state
                self.done_buf[idx] = done

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> Tuple[Any, ...]:
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        assert len(transition) == 5, "Inappropriate transition size"
        assert isinstance(transition[0], np.ndarray)
        assert isinstance(transition[1], np.ndarray)

        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, action = transition[:2]
            self._initialize_buffers(state, action)

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]

        self.obs_buf[self.idx] = curr_state
        self.acts_buf[self.idx] = action
        self.rews_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
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
        actions = self.acts_buf[indices]
        rewards = self.rews_buf[indices].reshape(-1, 1)
        next_states = self.next_obs_buf[indices]
        dones = self.done_buf[indices].reshape(-1, 1)

        return states, actions, rewards, next_states, dones

    def _initialize_buffers(self, state: np.ndarray, action: np.ndarray) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray
        self.obs_buf = np.zeros([self.max_len] + list(state.shape), dtype=state.dtype)
        self.acts_buf = np.zeros(
            [self.max_len] + list(action.shape), dtype=action.dtype
        )
        self.rews_buf = np.zeros([self.max_len], dtype=float)
        self.next_obs_buf = np.zeros(
            [self.max_len] + list(state.shape), dtype=state.dtype
        )
        self.done_buf = np.zeros([self.max_len], dtype=float)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length


class RecurrentReplayBuffer(BaseBuffer):
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
        sequence_size (int): sequence size for unrolling recurrent network
        overlap_size (int): overlapping size between the sequences
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(
        self,
        max_len: int,
        batch_size: int,
        sequence_size: int,
        overlap_size: int,
        gamma: float = 0.99,
        n_step: int = 1,
        demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
    ):
        """Initialize a ReplayBuffer object.
        Args:
            max_len (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            sequence_size (int): sequence size for unrolling recurrent network
            overlap_size (int): overlapping size between the sequences
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human play
        """
        assert 0 < batch_size <= max_len
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= max_len

        self.init_state = None
        self.init_action = None
        self.init_hidden = None

        self.local_obs_buf: np.ndarray = None
        self.local_acts_buf: np.ndarray = None
        self.local_rews_buf: np.ndarray = None
        self.local_hiddens_buf: torch.Tensor = None
        self.local_done_buf: np.ndarray = None

        self.obs_buf: np.ndarray = None
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.hiddens_buf: torch.Tensor = None
        self.done_buf: np.ndarray = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.max_len = max_len
        self.batch_size = batch_size
        self.overlap_size = overlap_size
        self.sequence_size = sequence_size
        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.episode_idx = self.demo_size
        self.idx = 0

    def add(
        self,
        transition: Tuple[
            np.ndarray, np.ndarray, torch.Tensor, float, np.ndarray, bool
        ],
    ) -> Tuple[Any, ...]:  # delete here
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        Add transitions to local buffer until it's full,
        and move thoese transitions to global buffer.
        """
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0 and self.idx == 0:
            state, action, hidden_state = transition[:3]
            self._initialize_buffers(state, action, hidden_state)

        # add a multi step transition
        reward, _, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action, hidden_state = self.n_step_buffer[0][:3]

        self.local_obs_buf[self.idx] = curr_state
        self.local_acts_buf[self.idx] = action
        self.local_rews_buf[self.idx] = reward
        self.local_hiddens_buf[self.idx] = hidden_state
        self.local_done_buf[self.idx] = done

        self.idx += 1
        if done and self.idx < self.sequence_size:
            self.idx = self.sequence_size

        if self.idx % self.sequence_size == 0:
            self.obs_buf[self.episode_idx] = self.local_obs_buf
            self.acts_buf[self.episode_idx] = self.local_acts_buf
            self.rews_buf[self.episode_idx] = self.local_rews_buf
            self.hiddens_buf[self.episode_idx] = self.local_hiddens_buf
            self.done_buf[self.episode_idx] = self.local_done_buf

            self.idx = self.overlap_size
            self.episode_idx += 1
            self._overlap_local_buffers()
            self.episode_idx = (
                0 if self.episode_idx % self.max_len == 0 else self.episode_idx
            )
            self.length = min(self.length + 1, self.max_len)

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def sample(self, indices: List[int] = None) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states = self.obs_buf[indices]
        actions = self.acts_buf[indices]
        rewards = self.rews_buf[indices]
        hidden_state = self.hiddens_buf[indices]
        dones = self.done_buf[indices]

        return states, actions, rewards, hidden_state, dones

    def _initialize_local_buffers(self):
        """Initialze local buffers for state, action, reward, hidden_state, done."""
        self.local_obs_buf = np.zeros(
            [self.sequence_size] + list(self.init_state.shape),
            dtype=self.init_state.dtype,
        )
        self.local_acts_buf = np.zeros(
            [self.sequence_size] + list(self.init_action.shape),
            dtype=self.init_action.dtype,
        )
        self.local_hiddens_buf = torch.zeros(
            [self.sequence_size] + list(self.init_hidden.shape),
            dtype=self.init_hidden.dtype,
        )
        self.local_rews_buf = np.zeros([self.sequence_size], dtype=float)

        self.local_done_buf = np.zeros([self.sequence_size], dtype=float)

    def _overlap_local_buffers(self):
        """Overlap the local buffers when the local buffers are full."""
        overlap_obs_buf = self.local_obs_buf[-self.overlap_size :]
        overlap_acts_buf = self.local_acts_buf[-self.overlap_size :]
        overlap_hiddens_buf = self.local_hiddens_buf[-self.overlap_size :]
        overlap_rews_buf = self.local_rews_buf[-self.overlap_size :]
        overlap_done_buf = self.local_done_buf[-self.overlap_size :]

        self._initialize_local_buffers()
        self.local_obs_buf[: self.overlap_size] = overlap_obs_buf
        self.local_acts_buf[: self.overlap_size] = overlap_acts_buf
        self.local_hiddens_buf[: self.overlap_size] = overlap_hiddens_buf
        self.local_rews_buf[: self.overlap_size] = overlap_rews_buf
        self.local_done_buf[: self.overlap_size] = overlap_done_buf

    def _initialize_buffers(
        self, state: np.ndarray, action: np.ndarray, hidden: torch.Tensor
    ) -> None:
        """Initialze global buffers for state, action, reward, hidden_state, done."""
        # In case action of demo is not np.ndarray
        self.init_state = state
        self.init_action = action
        self.init_hidden = hidden

        self.obs_buf = np.zeros(
            [self.max_len] + [self.sequence_size] + list(state.shape),
            dtype=state.dtype,
        )
        self.acts_buf = np.zeros(
            [self.max_len] + [self.sequence_size] + list(action.shape),
            dtype=action.dtype,
        )
        self.hiddens_buf = torch.zeros(
            [self.max_len] + [self.sequence_size] + list(hidden.shape),
            dtype=hidden.dtype,
        ).to(device)
        self.rews_buf = np.zeros([self.max_len] + [self.sequence_size], dtype=float)

        self.done_buf = np.zeros([self.max_len] + [self.sequence_size], dtype=float)

        self._initialize_local_buffers()

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length
