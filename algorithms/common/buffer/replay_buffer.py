# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
import torch

from algorithms.common.helper_functions import get_n_step_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py

    Attributes:
        obs_buf(np.ndarray): buffer of observations that will be initialized in _initialize()
        acts_buf(np.ndarray): buffer of observations that will be initialized in _initialize()
        rews_buf(np.ndarray): buffer of observations that will be initialized in _initialize()
        next_obs_buf(np.ndarray): buffer of observations that will be initialized in _initialize()
        done_buf(np.ndarray): buffer of observations that will be initialized in _initialize()
        buffer_size (int): size of each buffers
        batch_size (int): size of a batched sampled from replay buffer for training
        idx (int): index of current memory
        cur_size (int): amount of memory filled

    """

    def __init__(self, buffer_size: int, batch_size: int):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training

        """
        self.obs_buf: np.ndarray = None
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.next_obs_buf: np.ndarray = None
        self.done_buf: np.ndarray = None

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.idx = 0
        self.cur_size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ):
        """ Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        if self.cur_size == 0:
            self._initialize_buffers(state, action)

        self.obs_buf[self.idx] = state
        self.acts_buf[self.idx] = action
        self.rews_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
        self.done_buf[self.idx] = done

        self.idx = (self.idx + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

    def extend(self, transitions: list):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(*transition)

    def sample(self) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert self.cur_size >= self.batch_size

        indices = np.random.choice(self.cur_size, size=self.batch_size, replace=False)

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
        return self.cur_size


class NStepTransitionBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
        buffer (list): list of replay buffer
        buffer_size (int): buffer size not storing demos
        demo_size (int): size of a demo to permanently store in the buffer
        cursor (int): position to store next transition coming in

    """

    def __init__(self, buffer_size: int, n_step: int, gamma: float, demo: list = None):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            demo (list): demonstration transitions

        """
        assert buffer_size > 0

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.buffer_size = buffer_size
        self.buffer: list = list()
        self.n_step = n_step
        self.gamma = gamma
        self.demo_size = 0
        self.cursor = 0

        # if demo exists
        if demo:
            self.demo_size = len(demo)
            self.buffer.extend(demo)

        self.buffer.extend([None] * self.buffer_size)

    def add(self, transition: Tuple[np.ndarray, ...]) -> Tuple[Any, ...]:
        """Add a new transition to memory."""
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]
        new_transition = (curr_state, action, reward, next_state, done)

        # insert the new transition to buffer
        idx = self.demo_size + self.cursor
        self.buffer[idx] = new_transition
        self.cursor = (self.cursor + 1) % self.buffer_size

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def sample(self, indices: List[int]) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

        states_ = torch.FloatTensor(np.array(states)).to(device)
        actions_ = torch.FloatTensor(np.array(actions)).to(device)
        rewards_ = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states_ = torch.FloatTensor(np.array(next_states)).to(device)
        dones_ = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        if torch.cuda.is_available():
            states_ = states_.cuda(non_blocking=True)
            actions_ = actions_.cuda(non_blocking=True)
            rewards_ = rewards_.cuda(non_blocking=True)
            next_states_ = next_states_.cuda(non_blocking=True)
            dones_ = dones_.cuda(non_blocking=True)

        return states_, actions_, rewards_, next_states_, dones_
