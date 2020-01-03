# -*- coding: utf-8 -*-
"""DQN agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
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
import time
from typing import Tuple

import gym
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import wandb

from algorithms.common.abstract.agent import Agent
from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
from algorithms.common.buffer.replay_buffer import ReplayBuffer
import algorithms.common.helper_functions as common_utils
import algorithms.dqn.utils as dqn_utils
from algorithms.registry import AGENTS
from algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class DQNAgent(Agent):
    """DQN interacting with environment.

    Attribute:
        memory (PrioritizedReplayBuffer): replay memory
        dqn (nn.Module): actor model to select actions
        dqn_target (nn.Module): target actor model to select actions
        dqn_optim (Optimizer): optimizer for training actor
        params (dict): hyper-parameters
        beta (float): beta parameter for prioritized replay buffer
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step number
        episode_step (int): step number of the current episode
        epsilon (float): parameter for epsilon greedy policy
        i_episode (int): current episode number
        n_step_buffer (deque): n-size buffer to calculate n-step returns
        use_n_step (bool): whether or not to use n-step returns

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        params: ConfigDict,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            params (dict): hyper-parameters

        """
        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros(1)
        self.episode_step = 0
        self.total_step = 0
        self.i_episode = 0

        self.params = params
        self.gamma = params.gamma
        self.tau = params.tau
        self.batch_size = params.batch_size
        self.buffer_size = params.buffer_size
        self.update_starts_from = params.update_starts_from

        self.multiple_learn = params.multiple_learn
        self.train_freq = params.train_freq
        self.gradient_clip = params.gradient_clip
        self.n_step = params.n_step
        self.epsilon = params.max_epsilon

        self.w_n_step = params.w_n_step
        self.w_q_reg = params.w_q_reg

        self.per_alpha = params.per_alpha
        self.per_beta = params.per_beta
        self.per_eps = params.per_eps

        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.use_conv = len(self.state_dim) > 1
        self.use_n_step = self.n_step > 1
        self.use_dist_q = params.use_dist_q
        self.use_noisy_net = params.use_noisy_net

        self._initialize()
        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory for a single step
            self.memory = PrioritizedReplayBuffer(
                self.buffer_size, self.batch_size, alpha=self.per_alpha,
            )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(
                    self.buffer_size, n_step=self.n_step, gamma=self.gamma,
                )

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        fc_input_size = (
            self.network_cfg.fc_input_size if self.use_conv else self.state_dim[0]
        )

        if self.use_conv:
            # create FC
            fc_model = dqn_utils.get_fc_model(
                self.params,
                fc_input_size,
                self.action_dim,
                self.network_cfg.hidden_sizes,
            )
            fc_model2 = dqn_utils.get_fc_model(
                self.params,
                fc_input_size,
                self.action_dim,
                self.network_cfg.hidden_sizes,
            )

            # create CNN
            self.dqn = dqn_utils.get_cnn_model(self.use_dist_q, fc_model)
            self.dqn_target = dqn_utils.get_cnn_model(self.use_dist_q, fc_model2)

        else:
            # create FC
            self.dqn = dqn_utils.get_fc_model(
                self.params,
                fc_input_size,
                self.action_dim,
                self.network_cfg.hidden_sizes,
            )
            self.dqn_target = dqn_utils.get_fc_model(
                self.params,
                fc_input_size,
                self.action_dim,
                self.network_cfg.hidden_sizes,
            )

        self.dqn_target.load_state_dict(self.dqn.state_dict())

        # create optimizer
        self.dqn_optim = optim.Adam(
            self.dqn.parameters(),
            lr=self.optim_cfg.lr_dqn,
            weight_decay=self.optim_cfg.weight_decay,
            eps=self.optim_cfg.adam_eps,
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None and os.path.exists(self.args.load_from):
            self.load_params(self.args.load_from)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        # pylint: disable=comparison-with-callable
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        else:
            state = self._preprocess_state(state)
            selected_action = self.dqn(state).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

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
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)

    def _get_dqn_loss(
        self, experiences: Tuple[torch.Tensor, ...], gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return element-wise dqn loss and Q-values."""

        if self.use_dist_q == "IQN":
            return dqn_utils.calculate_iqn_loss(
                model=self.dqn,
                target_model=self.dqn_target,
                experiences=experiences,
                gamma=gamma,
                batch_size=self.batch_size,
                n_tau_samples=self.params.n_tau_samples,
                n_tau_prime_samples=self.params.n_tau_prime_samples,
                kappa=self.params.kappa,
            )
        elif self.use_dist_q == "C51":
            return dqn_utils.calculate_c51_loss(
                model=self.dqn,
                target_model=self.dqn_target,
                experiences=experiences,
                gamma=gamma,
                batch_size=self.batch_size,
                v_min=self.params.v_min,
                v_max=self.params.v_max,
                atom_size=self.params.atoms,
            )
        else:
            return dqn_utils.calculate_dqn_loss(
                model=self.dqn,
                target_model=self.dqn_target,
                experiences=experiences,
                gamma=gamma,
            )

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        # 1 step loss
        experiences_1 = self.memory.sample(self.per_beta)
        weights, indices = experiences_1[-3:-1]
        gamma = self.gamma
        dq_loss_element_wise, q_values = self._get_dqn_loss(experiences_1, gamma)
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.gamma ** self.n_step
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(
                experiences_n, gamma
            )

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += dq_loss_n_element_wise * self.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.w_q_reg

        # total loss
        loss = dq_loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        common_utils.soft_update(self.dqn, self.dqn_target, self.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.per_eps
        self.memory.update_priorities(indices, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        if self.use_noisy_net:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss.item(), q_values.mean().item()

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
        }

        Agent.save_params(self, params, n_episode)

    def write_log(self, i: int, loss: np.ndarray, score: float, avg_time_cost: float):
        """Write log about loss and score"""
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
            # wandb.watch([self.dqn], log="parameters")

        # pre-training if needed
        self.pretrain()

        max_epsilon, min_epsilon, epsilon_decay = (
            self.params.max_epsilon,
            self.params.min_epsilon,
            self.params.epsilon_decay,
        )

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

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.update_starts_from:
                    if self.total_step % self.train_freq == 0:
                        for _ in range(self.multiple_learn):
                            loss = self.update_model()
                            losses.append(loss)  # for logging

                    # decrease epsilon
                    self.epsilon = max(
                        self.epsilon - (max_epsilon - min_epsilon) * epsilon_decay,
                        min_epsilon,
                    )

                state = next_state
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                self.write_log(self.i_episode, avg_loss, score, avg_time_cost)

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.save_params(self.i_episode)
        self.interim_test()
