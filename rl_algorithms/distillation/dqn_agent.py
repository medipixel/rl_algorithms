# -*- coding: utf-8 -*-
"""DQN agent for episodic tasks in OpenAI Gym.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
         https://arxiv.org/pdf/1509.06461.pdf (Double DQN)
         https://arxiv.org/pdf/1511.05952.pdf (PER)
         https://arxiv.org/pdf/1511.06581.pdf (Dueling)
         https://arxiv.org/pdf/1706.10295.pdf (NoisyNet)
         https://arxiv.org/pdf/1707.06887.pdf (C51)
         https://arxiv.org/pdf/1710.02298.pdf (Rainbow)
         https://arxiv.org/pdf/1806.06923.pdf (IQN)
"""

import pickle
import time
from typing import Tuple

import numpy as np
import torch

from rl_algorithms.common.buffer.priortized_replay_buffer import DistillationPER
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.registry import AGENTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class DistillationDQN(DQNAgent):
    """DQN for policy distillation."""

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory for a single step
            self.memory = DistillationPER(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                alpha=self.hyper_params.per_alpha,
            )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        # pylint: disable=comparison-with-callable
        state = self._preprocess_state(state)
        q_values = self.dqn(state)
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        else:
            selected_action = q_values.argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action, q_values.squeeze().detach().cpu().numpy()

    def step(
        self, action: np.ndarray, q_values: np.ndarray
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )

            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition, q_values)

        return next_state, reward, done, info

    def _add_transition_to_memory(
        self, transition: Tuple[np.ndarray, ...], q_values: np.ndarray
    ):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition, q_values)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.dqn], log="parameters")

        # pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            self.episode_step = 0
            losses = list()
            done = False
            score = 0

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action, q_values = self.select_action(state)
                next_state, reward, done, _ = self.step(action, q_values)
                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            loss = self.update_model()
                            losses.append(loss)  # for logging

                    # decrease epsilon
                    self.epsilon = max(
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon)
                        * self.hyper_params.epsilon_decay,
                        self.min_epsilon,
                    )

                state = next_state
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.save_params(self.i_episode)

        with open("data/distillation_buffer.pkl", "wb") as f:
            pickle.dump(self.memory, f)
