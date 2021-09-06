# TODO : Move to common buffer
import random
from typing import Tuple

import numpy as np
import torch

from rl_algorithms.common.abstract.buffer import BaseBuffer


class ReplayMemory(BaseBuffer):
    """ReplayMemory for ACER.

    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rews_buf (np.ndarray): rewards
        probs_buf (np.ndarray): probability of actions
        done_buf (np.ndarray): dones
        max_len (int): size of buffers
        n_rollout (int): number of rollout
        num_in_buffer (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(self, buffer_size: int, n_rollout: int):
        """Initialize a ReplayBuffer object."""
        self.obs_buf = None
        self.acts_buf = None
        self.rews_buf = None
        self.probs_buf = None
        self.done_buf = None
        self.buffer_size = buffer_size
        self.idx = 0
        self.num_in_buffer = 0
        self.n_rollout = n_rollout

    def add(self, seq_data: list):
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        if self.num_in_buffer == 0:
            state, action, reward, prob, done_mask = seq_data[0]
            self._initialize_buffers(state, prob)

        self.idx = (self.idx + 1) % (self.buffer_size - 1)

        for i, transition in enumerate(seq_data):
            state, action, reward, prob, done_mask = transition
            self.obs_buf[self.idx][i] = state
            self.acts_buf[self.idx][i] = action
            self.rews_buf[self.idx][i] = reward
            self.probs_buf[self.idx][i] = prob
            self.done_buf[self.idx][i] = done_mask

        self.num_in_buffer += 1
        self.num_in_buffer = min(self.buffer_size - 1, self.num_in_buffer)

    def _initialize_buffers(self, state: np.ndarray, probs: np.ndarray):
        """Initialze buffers for state, action, reward, prob, done."""
        self.obs_buf = np.zeros(
            [self.buffer_size, self.n_rollout] + list(state.shape), dtype=state.dtype
        )
        self.acts_buf = np.zeros([self.buffer_size, self.n_rollout, 1], dtype=np.uint8)
        self.rews_buf = np.zeros(
            [self.buffer_size, self.n_rollout, 1], dtype=np.float64
        )
        self.probs_buf = np.zeros(
            [self.buffer_size, self.n_rollout] + list(probs.shape), dtype=probs.dtype
        )
        self.done_buf = np.zeros([self.buffer_size, self.n_rollout, 1])

    def sample(self, on_policy=False) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory.
        If on_policy, using last experience."""

        if on_policy:
            state = self.obs_buf[self.idx]
            action = self.acts_buf[self.idx]
            reward = self.rews_buf[self.idx]
            prob = self.probs_buf[self.idx]
            done = self.done_buf[self.idx]

        else:
            idx = random.randint(1, self.num_in_buffer)
            state = self.obs_buf[idx]
            action = self.acts_buf[idx]
            reward = self.rews_buf[idx]
            prob = self.probs_buf[idx]
            done = self.done_buf[idx]

        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        prob = torch.FloatTensor(prob)
        done = torch.FloatTensor(done)

        return state, action, reward, prob, done

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.num_in_buffer
