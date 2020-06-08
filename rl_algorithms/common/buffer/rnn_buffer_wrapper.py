from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
import torch

from rl_algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
from rl_algorithms.common.helper_functions import get_n_step_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RnnBufferWrapper:
    def __init__(
        self,
        replay_buffer,
        n_step: int = 1,
        sequence_size: int = 20,
        overlap_size: int = 10,
    ):
        self.n_step = n_step
        self.buffer = replay_buffer

        self.init_state = None
        self.init_action = None
        self.init_hidden = None

        self.local_obs_buf: np.ndarray = None
        self.local_next_obs_buf: np.ndarray = None
        self.local_acts_buf: np.ndarray = None
        self.local_rews_buf: np.ndarray = None
        self.local_hiddens_buf: torch.Tensor = None
        self.local_done_buf: np.ndarray = None

        self.hiddens_buf: torch.Tensor = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)

        self.sequence_size = sequence_size
        self.overlap_size = overlap_size

        self.overlap_size = overlap_size
        self.sequence_size = sequence_size
        self.length = 0
        self.idx = 0

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> Tuple[Any, ...]:
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
        reward, next_state, done = get_n_step_info(
            self.n_step_buffer, self.buffer.gamma
        )
        curr_state, action, hidden_state = self.n_step_buffer[0][:3]

        self.local_obs_buf[self.idx] = curr_state
        self.local_next_obs_buf[self.idx] = next_state
        self.local_acts_buf[self.idx] = action
        self.local_rews_buf[self.idx] = reward
        self.local_hiddens_buf[self.idx] = hidden_state
        self.local_done_buf[self.idx] = done

        self.idx += 1
        if done and self.idx < self.sequence_size:
            self.idx = self.sequence_size

        if self.idx % self.sequence_size == 0:
            self.hiddens_buf[self.buffer.idx] = self.local_hiddens_buf
            self.buffer.add(
                (
                    self.local_obs_buf,
                    self.local_acts_buf,
                    self.local_rews_buf,
                    self.local_next_obs_buf,
                    self.local_done_buf,
                )
            )

            self.idx = self.overlap_size

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def extend(
        self, transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
    ):
        self.buffer.extend(transitions)

    def sample(
        self, indices: List[int] = None, beta: float = 0.4
    ) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        if isinstance(self.buffer, PrioritizedReplayBuffer):
            (
                states,
                actions,
                rewards,
                _,
                dones,
                weights,
                indices,
                eps_d,
            ) = self.buffer.sample(beta)
            hidden_state = self.hiddens_buf[indices]
            if torch.cuda.is_available():
                hidden_state = hidden_state.cuda(non_blocking=True)

            return (
                states,
                actions,
                rewards,
                hidden_state,
                dones,
                1,
                weights,
                indices,
                eps_d,
            )

        else:
            assert len(self.buffer) >= self.buffer.batch_size

            if indices is None:
                indices = np.random.choice(
                    len(self), size=self.batch_size, replace=False
                )

            states, actions, rewards, _, dones = self.buffer.sample(indices)
            hidden_state = self.hiddens_buf[indices]

            if torch.cuda.is_available():
                states = states.cuda(non_blocking=True)
                actions = actions.cuda(non_blocking=True)
                rewards = rewards.cuda(non_blocking=True)
                hidden_state = hidden_state.cuda(non_blocking=True)
                dones = dones.cuda(non_blocking=True)

            return states, actions, rewards, hidden_state, dones

    def _initialize_buffers(
        self, state: np.ndarray, action: np.ndarray, hidden: torch.Tensor
    ) -> None:
        self.init_state = state
        self.init_action = action
        self.init_hidden = hidden
        self._initialize_local_buffers()
        self.buffer._initialize_buffers(
            self.local_obs_buf,
            self.local_acts_buf,
            self.local_rews_buf,
            self.local_done_buf,
        )
        self.hiddens_buf = torch.zeros(
            [self.buffer.buffer_size] + [self.sequence_size] + list(hidden.shape),
            dtype=hidden.dtype,
        ).to(device)

    def _initialize_local_buffers(self):
        """Initialze local buffers for state, action, resward, hidden_state, done."""
        self.local_obs_buf = np.zeros(
            [self.sequence_size] + list(self.init_state.shape),
            dtype=self.init_state.dtype,
        )
        self.local_next_obs_buf = np.zeros(
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
        overlap_next_obs_buf = self.local_next_obs_buf[-self.overlap_size :]
        overlap_acts_buf = self.local_acts_buf[-self.overlap_size :]
        overlap_hiddens_buf = self.local_hiddens_buf[-self.overlap_size :]
        overlap_rews_buf = self.local_rews_buf[-self.overlap_size :]
        overlap_done_buf = self.local_done_buf[-self.overlap_size :]

        self._initialize_local_buffers()
        self.local_obs_buf[: self.overlap_size] = overlap_obs_buf
        self.local_next_obs_buf[: self.overlap_size] = overlap_next_obs_buf
        self.local_acts_buf[: self.overlap_size] = overlap_acts_buf
        self.local_hiddens_buf[: self.overlap_size] = overlap_hiddens_buf
        self.local_rews_buf[: self.overlap_size] = overlap_rews_buf
        self.local_done_buf[: self.overlap_size] = overlap_done_buf

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.buffer.length

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        self.buffer.update_priorities(indices, priorities)
