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
import os
import pickle
import random
import time
from typing import Tuple

import gym
import numpy as np
import torch
from tqdm import tqdm
import wandb

from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.registry import AGENTS, build_learner
from rl_algorithms.utils.config import ConfigDict


@AGENTS.register_module
class OfflineDQNAgent(Agent):
    """DQN interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
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
        Agent.__init__(self, env, env_info, args, log_cfg)

        self.curr_state = np.zeros(1)
        self.episode_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg
        self.learner_cfg.args = self.args
        self.learner_cfg.env_info = self.env_info
        self.learner_cfg.hyper_params = self.hyper_params
        self.learner_cfg.log_cfg = self.log_cfg

        self.per_beta = hyper_params.per_beta

        if self.learner_cfg.head.configs.use_noisy_net:
            self.max_epsilon = 0.0
            self.min_epsilon = 0.0
            self.epsilon = 0.0
        else:
            self.max_epsilon = hyper_params.max_epsilon
            self.min_epsilon = hyper_params.min_epsilon
            self.epsilon = hyper_params.max_epsilon

        self._initialize()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            self.load_offline_data()

        self.learner = build_learner(self.learner_cfg)

    def load_offline_data(self):
        total_dataset_num = 0
        file_name_list = []
        for _dir in self.hyper_params.dataset_path:
            total_dataset_num += len(os.listdir(_dir))
        # replay memory for a single step
        self.memory = ReplayBuffer(
            total_dataset_num,
            self.hyper_params.batch_size,
            n_step=self.hyper_params.n_step,
            gamma=self.hyper_params.gamma,
        )
        for _dir in self.hyper_params.dataset_path:
            tmp = os.listdir(_dir)
            tmp = sorted(tmp, key=lambda x: int(x.split(".")[0]))
            total_dataset_num += len(tmp)
            file_name_list += [os.path.join(_dir, x) for x in tmp]
        for file_name in tqdm(file_name_list):
            with open(file_name, "rb") as f:
                transition = pickle.load(f)
                self._add_transition_to_memory(transition)

    # pylint: disable=no-self-use
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        state = numpy2floattensor(state, self.learner.device)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""

        return self.env.step(action)

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        self.memory.add(transition)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, avg_time_cost = log_value
        print(
            "[INFO] episode %d, total step: %d, total score: %f\n"
            "epsilon: %f, loss: %f, avg q-value: %f (spent %.6f sec/step)\n"
            % (
                i,
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

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state
        with torch.no_grad():
            state = self._preprocess_state(state)
            selected_action = self.learner.dqn(state).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def sample_experience(self, idx) -> Tuple[torch.Tensor, ...]:
        """Sample experience from replay buffer."""
        experiences_1 = self.memory.sample(idx)
        experiences_1 = numpy2floattensor(experiences_1, self.learner.device)
        return experiences_1

    def make_offline_dir(self):
        """Make directory for saving offline data."""
        self.save_offline_dir = os.path.join(
            self.hyper_params.save_dir,
            "offline_data/"
            + self.env_info.name
            + "/"
            + self.log_cfg.curr_time
            + "/"
            + "offline/",
        )
        self.save_distillation_dir = os.path.join(
            self.hyper_params.save_dir,
            "offline_data/"
            + self.env_info.name
            + "/"
            + self.log_cfg.curr_time
            + "/"
            + "distillation/",
        )
        os.makedirs(self.save_offline_dir)
        os.makedirs(self.save_distillation_dir)
        self.save_count = 0

    def _save_offline_data(self, transition, q):
        distill_current_ep_dir = (
            f"{self.save_distillation_dir}/{self.save_count:07}.pkl"
        )
        offline_current_ep_dir = f"{self.save_offline_dir}/{self.save_count:07}.pkl"
        self.save_count += 1
        with open(distill_current_ep_dir, "wb") as f:
            pickle.dump([self.curr_state, q], f, protocol=pickle.HIGHEST_PROTOCOL)
        if transition:
            with open(offline_current_ep_dir, "wb") as f:
                pickle.dump(transition, f, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
        memory_len_range = [*range(len(self.memory))]
        iter_num = (len(self.memory) // self.hyper_params.batch_size) + 1
        for self.i_episode in range(1, self.args.episode_num + 1):
            losses = list()
            t_begin = time.time()
            for _ in tqdm(range(iter_num)):
                for _ in range(self.hyper_params.multiple_update):
                    idx = random.sample(memory_len_range, self.hyper_params.batch_size)
                    experience = self.sample_experience(idx)

                    info = self.learner.update_model(
                        experience, isinstance(self.memory, PrioritizedBufferWrapper)
                    )
                    loss = info[0:2]
                    losses.append(loss)  # for logging

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.i_episode

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, 0, avg_time_cost)
                self.write_log(log_value)

            self.learner.save_params(self.i_episode)
            # self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        # self.interim_test()
