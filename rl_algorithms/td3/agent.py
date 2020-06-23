# -*- coding: utf-8 -*-
"""TD3 agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1802.09477.pdf
"""

import argparse
import time
from typing import Tuple

import gym
import numpy as np
import torch
import wandb

from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.common.noise import GaussianNoise
from rl_algorithms.registry import AGENTS, build_learner
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class TD3Agent(Agent):
    """TD3 interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (ReplayBuffer): replay memory
        exploration_noise (GaussianNoise): random noise for exploration
        curr_state (np.ndarray): temporary storage of the current state
        total_steps (int): total step numbers
        episode_steps (int): step number of the current episode
        i_episode (int): current episode number
        noise_cfg (ConfigDict): config of noise

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
        noise_cfg: ConfigDict,
        log_cfg: ConfigDict,
    ):
        """Initialize.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        Agent.__init__(self, env, env_info, args, log_cfg)

        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.episode_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg
        self.learner_cfg.args = self.args
        self.learner_cfg.env_info = self.env_info
        self.learner_cfg.hyper_params = self.hyper_params
        self.learner_cfg.log_cfg = self.log_cfg
        self.learner_cfg.noise_cfg = noise_cfg
        self.learner_cfg.device = device

        # noise instance to make randomness of action
        self.exploration_noise = GaussianNoise(
            self.env_info.action_space.shape[0],
            noise_cfg.exploration_noise,
            noise_cfg.exploration_noise,
        )

        if not self.args.test:
            # replay memory
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size
            )

        self.learner = build_learner(self.learner_cfg)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        # initial training step, try random action for exploration
        self.curr_state = state

        if (
            self.total_step < self.hyper_params.initial_random_action
            and not self.args.test
        ):
            return np.array(self.env_info.action_space.sample())

        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            selected_action = self.learner.actor(state).detach().cpu().numpy()

        if not self.args.test:
            noise = self.exploration_noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            # if last state is not terminal state in episode, done is false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            self.memory.add((self.curr_state, action, reward, next_state, done_bool))

        return next_state, reward, done, info

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, policy_update_freq, avg_time_cost = log_value
        total_loss = loss.sum()
        print(
            "[INFO] episode %d, episode_step: %d, total_step: %d, total score: %d\n"
            "total loss: %f actor_loss: %.3f critic1_loss: %.3f critic2_loss: %.3f "
            "(spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0] * policy_update_freq,  # actor loss
                loss[1],  # critic1 loss
                loss[2],  # critic2 loss
                avg_time_cost,
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * policy_update_freq,
                    "critic1 loss": loss[1],
                    "critic2 loss": loss[2],
                    "time per each step": avg_time_cost,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.actor, self.critic1, self.critic2], log="parameters")

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()
            self.episode_step = 0

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                state = next_state
                score += reward

                if len(self.memory) >= self.hyper_params.batch_size:
                    experience = self.memory.sample()
                    experience = numpy2floattensor(experience)
                    loss = self.learner.update_model(experience)
                    loss_episode.append(loss)  # for logging

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                log_value = (
                    self.i_episode,
                    avg_loss,
                    score,
                    self.hyper_params.policy_update_freq,
                    avg_time_cost,
                )
                self.write_log(log_value)
            if self.i_episode % self.args.save_period == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()
