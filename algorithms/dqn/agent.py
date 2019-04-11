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
import wandb

from algorithms.common.abstract.agent import Agent
from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
from algorithms.common.buffer.replay_buffer import NStepTransitionBuffer
import algorithms.common.helper_functions as common_utils
import algorithms.dqn.utils as dqn_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
    """DQN interacting with environment.

    Attribute:
        memory (PrioritizedReplayBuffer): replay memory
        dqn (nn.Module): actor model to select actions
        dqn_target (nn.Module): target actor model to select actions
        dqn_optimizer (Optimizer): optimizer for training actor
        hyper_params (dict): hyper-parameters
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
        hyper_params: dict,
        models: tuple,
        optim: torch.optim.Adam,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including main network and target
            optim (torch.optim.Adam): optimizers for dqn

        """
        Agent.__init__(self, env, args)

        self.use_n_step = hyper_params["N_STEP"] > 1
        self.epsilon = hyper_params["MAX_EPSILON"]
        self.dqn, self.dqn_target = models
        self.hyper_params = hyper_params
        self.curr_state = np.zeros(1)
        self.dqn_optimizer = optim
        self.episode_step = 0
        self.total_step = 0
        self.i_episode = 0

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        self._initialize()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory for a single step
            self.beta = self.hyper_params["PER_BETA"]
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params["BUFFER_SIZE"],
                self.hyper_params["BATCH_SIZE"],
                alpha=self.hyper_params["PER_ALPHA"],
            )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = NStepTransitionBuffer(
                    self.hyper_params["BUFFER_SIZE"],
                    n_step=self.hyper_params["N_STEP"],
                    gamma=self.hyper_params["GAMMA"],
                )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        # pylint: disable=comparison-with-callable
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        self.total_step += 1
        self.episode_step += 1

        next_state, reward, done, _ = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )

            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(*transition)

    def _get_dqn_loss(
        self, experiences: Tuple[torch.Tensor, ...], gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return element-wise dqn loss and Q-values."""

        if self.hyper_params["USE_DIST_Q"] == "IQN":
            return dqn_utils.calculate_iqn_loss(
                model=self.dqn,
                target_model=self.dqn_target,
                experiences=experiences,
                gamma=gamma,
                batch_size=self.hyper_params["BATCH_SIZE"],
                n_tau_samples=self.hyper_params["N_TAU_SAMPLES"],
                n_tau_prime_samples=self.hyper_params["N_TAU_PRIME_SAMPLES"],
                kappa=self.hyper_params["KAPPA"],
            )
        elif self.hyper_params["USE_DIST_Q"] == "C51":
            return dqn_utils.calculate_c51_loss(
                model=self.dqn,
                target_model=self.dqn_target,
                experiences=experiences,
                gamma=gamma,
                batch_size=self.hyper_params["BATCH_SIZE"],
                v_min=self.hyper_params["V_MIN"],
                v_max=self.hyper_params["V_MAX"],
                atom_size=self.hyper_params["ATOMS"],
            )
        else:
            return dqn_utils.calculate_dqn_loss(
                model=self.dqn,
                target_model=self.dqn_target,
                experiences=experiences,
                gamma=gamma,
            )

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        # 1 step loss
        experiences_1 = self.memory.sample(self.beta)
        weights, indices = experiences_1[-2:]
        gamma = self.hyper_params["GAMMA"]
        dq_loss_element_wise, q_values = self._get_dqn_loss(experiences_1, gamma)
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.hyper_params["GAMMA"] ** self.hyper_params["N_STEP"]
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(
                experiences_n, gamma
            )

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += (
                dq_loss_n_element_wise * self.hyper_params["W_N_STEP"]
            )
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params["W_Q_REG"]

        # total loss
        loss = dq_loss + q_regular

        self.dqn_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params["GRADIENT_CLIP"])
        self.dqn_optimizer.step()

        # update target networks
        tau = self.hyper_params["TAU"]
        common_utils.soft_update(self.dqn, self.dqn_target, tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params["PER_EPS"]
        self.memory.update_priorities(indices, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        if self.hyper_params["USE_NOISY_NET"]:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return loss.data, q_values.mean().data

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optimizer.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optimizer.state_dict(),
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
            wandb.init()
            wandb.config.update(self.hyper_params)
            # wandb.watch([self.dqn], log="parameters")

        # pre-training if needed
        self.pretrain()

        max_epsilon, min_epsilon, epsilon_decay = (
            self.hyper_params["MAX_EPSILON"],
            self.hyper_params["MIN_EPSILON"],
            self.hyper_params["EPSILON_DECAY"],
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
                next_state, reward, done = self.step(action)

                if len(self.memory) >= self.hyper_params["UPDATE_STARTS_FROM"]:
                    if self.total_step % self.hyper_params["TRAIN_FREQ"] == 0:
                        for _ in range(self.hyper_params["MULTIPLE_LEARN"]):
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
