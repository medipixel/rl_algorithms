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
import time
from typing import Tuple

import gym
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import wandb

from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
import rl_algorithms.dqn.utils as dqn_utils
from rl_algorithms.registry import AGENTS
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class DQNAgent(Agent):
    """DQN interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (PrioritizedReplayBuffer): replay memory
        dqn (nn.Module): actor model to select actions
        dqn_target (nn.Module): target actor model to select actions
        dqn_optim (Optimizer): optimizer for training actor
        curr_state (np.ndarray): temporary storage of the current state
        total_step (int): total step number
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        epsilon (float): parameter for epsilon greedy policy
        n_step_buffer (deque): n-size buffer to calculate n-step returns
        per_beta (float): beta parameter for prioritized replay buffer
        use_conv (bool): whether or not to use convolution layer
        use_n_step (bool): whether or not to use n-step returns

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
    ):
        """Initialize."""
        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros(1)
        self.episode_step = 0
        self.total_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.per_beta = hyper_params.per_beta
        self.use_conv = len(self.state_dim) > 1
        self.use_n_step = hyper_params.n_step > 1

        if hyper_params.use_noisy_net:
            self.max_epsilon = 0.0
            self.min_epsilon = 0.0
            self.epsilon = 0.0
        else:
            self.max_epsilon = hyper_params.max_epsilon
            self.min_epsilon = hyper_params.min_epsilon
            self.epsilon = hyper_params.max_epsilon

        self._initialize()
        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory for a single step
            self.memory = PrioritizedReplayBuffer(
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

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""

        if self.use_conv:
            # create CNN
            self.dqn = dqn_utils.get_cnn_model(
                self.hyper_params, self.action_dim, self.state_dim, self.network_cfg
            )
            self.dqn_target = dqn_utils.get_cnn_model(
                self.hyper_params, self.action_dim, self.state_dim, self.network_cfg
            )

        else:
            # create FC
            fc_input_size = self.state_dim[0]

            self.dqn = dqn_utils.get_fc_model(
                self.hyper_params,
                fc_input_size,
                self.action_dim,
                self.network_cfg.hidden_sizes,
            )
            self.dqn_target = dqn_utils.get_fc_model(
                self.hyper_params,
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
        if self.args.load_from is not None:
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

        if self.hyper_params.use_dist_q == "IQN":
            return dqn_utils.calculate_iqn_loss(
                model=self.dqn,
                target_model=self.dqn_target,
                experiences=experiences,
                gamma=gamma,
                batch_size=self.hyper_params.batch_size,
                n_tau_samples=self.hyper_params.n_tau_samples,
                n_tau_prime_samples=self.hyper_params.n_tau_prime_samples,
                kappa=self.hyper_params.kappa,
            )
        elif self.hyper_params.use_dist_q == "C51":
            return dqn_utils.calculate_c51_loss(
                model=self.dqn,
                target_model=self.dqn_target,
                experiences=experiences,
                gamma=gamma,
                batch_size=self.hyper_params.batch_size,
                v_min=self.hyper_params.v_min,
                v_max=self.hyper_params.v_max,
                atom_size=self.hyper_params.atoms,
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
        gamma = self.hyper_params.gamma
        dq_loss_element_wise, q_values = self._get_dqn_loss(experiences_1, gamma)
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(
                experiences_n, gamma
            )

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params.w_q_reg

        # total loss
        loss = dq_loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params.per_eps
        self.memory.update_priorities(indices, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        if self.hyper_params.use_noisy_net:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss.item(), q_values.mean().item()

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):  # type: ignore
        """Save model and optimizer parameters."""
        params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
        }

        Agent.save_params(self, params, n_episode)

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
        self.interim_test()
