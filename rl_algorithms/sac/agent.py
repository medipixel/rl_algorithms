# -*- coding: utf-8 -*-
"""SAC agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
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
from rl_algorithms.registry import AGENTS
from rl_algorithms.sac.learner import SACLearner
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class SACAgent(Agent):
    """SAC agent interacting with environment.

    Attrtibutes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (ReplayBuffer): replay memory
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        update_step (int): step number of updates
        i_episode (int): current episode number

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
        optim_cfg: ConfigDict,
    ):
        """Initialize.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.episode_step = 0
        self.update_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.backbone_cfg = backbone
        self.head_cfg = head
        self.optim_cfg = optim_cfg

        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]
        self.head_cfg["action_dim"] = self.action_dim
        self.head_cfg.actor.configs.state_size = (
            self.head_cfg.critic_vf.configs.state_size
        ) = self.state_dim
        self.head_cfg.critic_qf.configs.state_size = (
            self.state_dim[0] + self.action_dim,
        )
        self.head_cfg.actor.configs.output_size = self.action_dim

        self._initialize()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size
            )

        self.learner = SACLearner(
            self.args,
            self.hyper_params,
            self.log_cfg,
            self.head_cfg,
            self.backbone_cfg,
            self.optim_cfg,
            device,
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state(state)

        # if initial random action should be conducted
        if (
            self.total_step < self.hyper_params.initial_random_action
            and not self.args.test
        ):
            return np.array(self.env.action_space.sample())

        if self.args.test:
            _, _, _, selected_action, _ = self.learner.actor(state)
        else:
            selected_action, _, _, _, _ = self.learner.actor(state)

        return selected_action.detach().cpu().numpy()

    # pylint: disable=no-self-use
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        state = torch.FloatTensor(state).to(device)
        return state

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        self.memory.add(transition)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, policy_update_freq, avg_time_cost = log_value
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "total loss: %.3f actor_loss: %.3f qf_1_loss: %.3f qf_2_loss: %.3f "
            "vf_loss: %.3f alpha_loss: %.3f (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0] * policy_update_freq,  # actor loss
                loss[1],  # qf_1 loss
                loss[2],  # qf_2 loss
                loss[3],  # vf loss
                loss[4],  # alpha loss
                avg_time_cost,
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * policy_update_freq,
                    "qf_1 loss": loss[1],
                    "qf_2 loss": loss[2],
                    "vf loss": loss[3],
                    "alpha loss": loss[4],
                    "time per each step": avg_time_cost,
                }
            )

    # pylint: disable=no-self-use, unnecessary-pass
    def pretrain(self):
        """Pretraining steps."""
        pass

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.actor, self.vf, self.qf_1, self.qf_2], log="parameters")

        # pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            loss_episode = list()

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

                # training
                if len(self.memory) >= self.hyper_params.batch_size:
                    for _ in range(self.hyper_params.multiple_update):
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
