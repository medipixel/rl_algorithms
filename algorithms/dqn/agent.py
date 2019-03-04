# -*- coding: utf-8 -*-
"""DQN agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
         https://arxiv.org/pdf/1509.06461.pdf (Double DQN)
         https://arxiv.org/pdf/1511.05952.pdf (PER)
         https://arxiv.org/pdf/1511.06581.pdf (Dueling)
"""

import argparse
import datetime
import os
from typing import Tuple

import gym
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import wandb

from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
from algorithms.common.env.multiprocessing_env import SubprocVecEnv
import algorithms.common.helper_functions as common_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """DQN interacting with environment.

    Attribute:
        memory (PrioritizedReplayBuffer): replay memory
        dqn (nn.Module): actor model to select actions
        dqn_target (nn.Module): target actor model to select actions
        dqn_optimizer (Optimizer): optimizer for training actor
        hyper_params (dict): hyper-parameters
        beta (float): beta parameter for prioritized replay buffer
        curr_state (np.ndarray): temporary storage of the current state
        total_steps (np.ndarray): total step numbers
        episode_steps (np.ndarray): step number of the current episode
        epsilon (float): parameter for epsilon greedy policy
        i_episode (int): current episode number

    """

    def __init__(
        self,
        env_single: gym.Env,
        env_multi: SubprocVecEnv,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optim: torch.optim.Adam,
    ):
        """Initialization.

        Args:
            env_single (gym.Env): openAI Gym environment
            env_multi (SubprocVecEnv): Gym env with multiprocessing for training
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including main network and target
            optim (torch.optim.Adam): optimizers for dqn

        """
        AbstractAgent.__init__(self, env_single, args)

        if not self.args.test:
            self.env = env_multi
        self.dqn, self.dqn_target = models
        self.dqn_optimizer = optim
        self.hyper_params = hyper_params
        self.curr_state = np.zeros((1,))
        self.total_steps = np.zeros(hyper_params["N_WORKERS"], dtype=np.int)
        self.episode_steps = np.zeros(hyper_params["N_WORKERS"], dtype=np.int)
        self.epsilon = self.hyper_params["MAX_EPSILON"]
        self.i_episode = 0

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        self._initialize()

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory
            self.beta = self.hyper_params["PER_BETA"]
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params["BUFFER_SIZE"],
                self.hyper_params["BATCH_SIZE"],
                alpha=self.hyper_params["PER_ALPHA"],
            )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        # pylint: disable=comparison-with-callable
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = self.env.sample()
        else:
            state = torch.FloatTensor(state).to(device)
            selected_action = self.dqn(state, self.epsilon).argmax(dim=-1)
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        self.total_steps += 1
        self.episode_steps += 1

        next_state, reward, done, _ = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = done.copy()
            done_bool[
                np.where(self.episode_steps == self.args.max_episode_steps)
            ] = False

            action = action.tolist()
            reward = reward.tolist()
            done_bool = done_bool.tolist()

            for s, a, r, n_s, d in zip(
                self.curr_state, action, reward, next_state, done_bool
            ):
                self.memory.add(s, a, r, n_s, d)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        experiences = self.memory.sample(self.beta)
        states, actions, rewards, next_states, dones, weights, indexes = experiences

        q_values = self.dqn(states, self.epsilon)
        next_q_values = self.dqn(next_states, self.epsilon)
        next_target_q_values = self.dqn_target(next_states, self.epsilon)

        curr_q_value = q_values.gather(1, actions.long().unsqueeze(1))
        next_q_value = next_target_q_values.gather(  # Double DQN
            1, next_q_values.argmax(1).unsqueeze(1)
        )

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        target = rewards + self.hyper_params["GAMMA"] * next_q_value * masks
        target = target.to(device)

        # calculate dq loss
        dq_loss_element_wise = (target - curr_q_value).pow(2)
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
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy().squeeze()
        new_priorities = loss_for_prior + self.hyper_params["PER_EPS"]
        self.memory.update_priorities(indexes, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        return loss.data

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

        AbstractAgent.save_params(self, params, n_episode)

    def write_log(self, i: int, loss: np.ndarray, score: int):
        """Write log about loss and score"""
        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "epsilon: %f, loss: %f, at %s\n"
            % (
                i,
                self.episode_steps[0],
                self.total_steps.sum(),
                score,
                self.epsilon,
                loss,
                datetime.datetime.now(),
            )
        )

        if self.args.log:
            wandb.log({"score": score, "dqn loss": loss, "epsilon": self.epsilon})

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

        state = self.env.reset()
        i_episode_prev = 0
        losses = list()
        i_episode = 0
        score = 0

        while i_episode <= self.args.episode_num:
            if self.args.render and i_episode >= self.args.render_after:
                self.env.render()

            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward[0]
            i_episode_prev = i_episode
            i_episode += done.sum()
            self.i_episode = i_episode

            if (i_episode // self.args.save_period) != (
                i_episode_prev // self.args.save_period
            ):
                self.save_params(i_episode)

            if done[0]:
                if losses:
                    avg_loss = np.vstack(losses).mean(axis=0)
                    self.write_log(i_episode, avg_loss, score)
                    losses.clear()
                score = 0

            self.episode_steps[np.where(done)] = 0

            if len(self.memory) >= self.hyper_params["UPDATE_STARTS_FROM"]:
                for _ in range(self.hyper_params["MULTIPLE_LEARN"]):
                    loss = self.update_model()
                    losses.append(loss)  # for logging

                # decrease epsilon
                max_epsilon, min_epsilon, epsilon_decay, n_workers = (
                    self.hyper_params["MAX_EPSILON"],
                    self.hyper_params["MIN_EPSILON"],
                    self.hyper_params["EPSILON_DECAY"],
                    self.hyper_params["N_WORKERS"],
                )
                self.epsilon = max(
                    self.epsilon
                    - (max_epsilon - min_epsilon) * epsilon_decay * n_workers,
                    min_epsilon,
                )

        # termination
        self.env.close()
        self.save_params(i_episode)
