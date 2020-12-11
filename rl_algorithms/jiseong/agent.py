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

import argparse
from collections import deque
import time
from typing import Tuple

import gym
import numpy as np
import torch
import wandb

from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.jiseong.buffer import JBuffer
from rl_algorithms.registry import AGENTS, build_learner
from rl_algorithms.utils.config import ConfigDict


@AGENTS.register_module
class JAgent(DQNAgent):
    """DQN interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (PrioritizedReplayBuffer): replay memory
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step number
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        epsilon (float): parameter for epsilon greedy policy
        n_step_buffer (deque): n-size buffer to calculate n-step returns
        per_beta (float): beta parameter for prioritized replay buffer

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
        log_cfg: ConfigDict,
    ):
        """Initialize."""
        DQNAgent.__init__(self, env, env_info, args, hyper_params, learner_cfg, log_cfg)
        self.state_info = deque([], maxlen=hyper_params.info_len)

        self._initialize()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory for a single step
            self.memory = JBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size,
            )
            self.memory = PrioritizedBufferWrapper(
                self.memory, alpha=self.hyper_params.per_alpha
            )

        self.learner = build_learner(self.learner_cfg)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state = self._preprocess_state(state)
                selected_action = self.learner.dqn(state, self.state_info).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    # pylint: disable=no-self-use
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        state = numpy2floattensor(state, self.learner.device)
        info_tensor = numpy2floattensor(np.array(self.state_info), self.learner.device)
        return state, info_tensor

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    @staticmethod
    def get_state_info(info):
        return np.array(info)

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        # add a single step transition
        # if transition is not an empty tuple
        self.memory.add(transition)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, avg_time_cost = log_value
        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %f\n"
            "epsilon: %f, loss: %f, avg q-value: %f (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                self.epsilon,
                loss[0],
                loss[1],
                avg_time_cost,
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "epsilon": self.epsilon,
                    "dqn loss": loss[0],
                    "avg q values": loss[1],
                    "time per each step": avg_time_cost,
                    "total_step": self.total_step,
                }
            )

    def sample_experience(self) -> Tuple[torch.Tensor, ...]:
        """Sample experience from replay buffer."""
        experiences_1 = self.memory.sample(self.per_beta)
        experiences_1 = (
            numpy2floattensor(experiences_1[:8], self.learner.device)
            + experiences_1[8:]
        )
        return experiences_1

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
                state_info = self.state_info.copy()
                action = self.select_action(state)
                next_state, reward, done, info = self.step(action)
                next_state_info = self.get_state_info(info)
                self.state_info.append(next_state_info)
                if not self.args.test:
                    # if the last state is not a terminal state, store done as false
                    done_bool = (
                        False
                        if self.episode_step == self.args.max_episode_steps
                        else done
                    )

                    transition = (
                        self.curr_state,
                        np.array(state_info),
                        action,
                        reward,
                        next_state,
                        np.array(self.state_info),
                        done_bool,
                    )
                    self._add_transition_to_memory(transition)

                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            experience = self.sample_experience()
                            info = self.learner.update_model(experience)
                            loss = info[0:2]
                            indices, new_priorities = info[2:4]
                            losses.append(loss)  # for logging
                            self.memory.update_priorities(indices, new_priorities)

                    # decrease epsilon
                    self.epsilon = max(
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon)
                        * self.hyper_params.epsilon_decay,
                        self.min_epsilon,
                    )

                    # increase priority beta
                    fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
                    self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

                state = next_state
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()
