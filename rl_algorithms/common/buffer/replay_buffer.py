# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
import torch

from rl_algorithms.common.helper_functions import get_n_step_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
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
        buffer_size (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        gamma: float = 0.99,
        n_step: int = 1,
        demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
    ):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human play
        """
        assert 0 < batch_size <= buffer_size
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= buffer_size

        self.obs_buf: np.ndarray = None
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.next_obs_buf: np.ndarray = None
        self.done_buf: np.ndarray = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size

        # demo may have empty tuple list [()]
        if self.demo and self.demo[0]:
            self.buffer_size += self.demo_size
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
        self.idx = self.demo_size if self.idx % self.buffer_size == 0 else self.idx
        self.length = min(self.length + 1, self.buffer_size)

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def extend(
        self, transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
    ):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(transition)

    def sample(self, indices: List[int] = None) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states = torch.FloatTensor(self.obs_buf[indices]).to(device)
        actions = torch.FloatTensor(self.acts_buf[indices]).to(device)
        rewards = torch.FloatTensor(self.rews_buf[indices].reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(self.next_obs_buf[indices]).to(device)
        dones = torch.FloatTensor(self.done_buf[indices].reshape(-1, 1)).to(device)

        if torch.cuda.is_available():
            states = states.cuda(non_blocking=True)
            actions = actions.cuda(non_blocking=True)
            rewards = rewards.cuda(non_blocking=True)
            next_states = next_states.cuda(non_blocking=True)
            dones = dones.cuda(non_blocking=True)

        return states, actions, rewards, next_states, dones

    def _initialize_buffers(self, state: np.ndarray, action: np.ndarray) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray
        self.obs_buf = np.zeros(
            [self.buffer_size] + list(state.shape), dtype=state.dtype
        )
        self.acts_buf = np.zeros(
            [self.buffer_size] + list(action.shape), dtype=action.dtype
        )
        self.rews_buf = np.zeros([self.buffer_size], dtype=float)
        self.next_obs_buf = np.zeros(
            [self.buffer_size] + list(state.shape), dtype=state.dtype
        )
        self.done_buf = np.zeros([self.buffer_size], dtype=float)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length
