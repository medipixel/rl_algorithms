# -*- coding: utf-8 -*-
"""DQN agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
         https://arxiv.org/pdf/1509.06461.pdf
         https://arxiv.org/pdf/1511.05952.pdf
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import wandb

import algorithms.common.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer

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
        total_step (int): total step numbers
        episode_step (int): step number of the current episode
        epsilon (float): parameter for epsilon greedy policy

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
        AbstractAgent.__init__(self, env, args)

        self.dqn, self.dqn_target = models
        self.dqn_optimizer = optim
        self.hyper_params = hyper_params
        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.episode_step = 0
        self.epsilon = self.hyper_params["MAX_EPSILON"] if not self.args.test else 0

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

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

        max_epsilon, min_epsilon, epsilon_decay = (
            self.hyper_params["MAX_EPSILON"],
            self.hyper_params["MIN_EPSILON"],
            self.hyper_params["EPSILON_DECAY"],
        )

        # decrease epsilon
        self.epsilon = max(
            self.epsilon - (max_epsilon - min_epsilon) * epsilon_decay, min_epsilon
        )

        # epsilon greedy policy
        if not self.args.test and self.epsilon > np.random.random():  # random action
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).to(device)
            selected_action = self.dqn(state).argmax()
            return selected_action.detach().cpu().numpy()

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
            self.memory.add(self.curr_state, action, reward, next_state, done_bool)

        return next_state, reward, done

    def update_model(self, experiences: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones, weights, indexes = experiences

        q_values = self.dqn(states)
        next_q_values = self.dqn(next_states)
        next_target_q_values = self.dqn_target(next_states)

        curr_q_value = q_values.gather(1, actions.long().unsqueeze(1))
        next_q_value = next_target_q_values.gather(
            1, next_q_values.argmax(1).unsqueeze(1)
        )

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        target = rewards + self.hyper_params["GAMMA"] * next_q_value * masks
        target = target.to(device)

        loss = torch.mean((target - curr_q_value).pow(2) * weights)

        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()

        # update target networks
        tau = self.hyper_params["TAU"]
        common_utils.soft_update(self.dqn, self.dqn_target, tau)

        # update priorities in PER
        new_priorities = (target - curr_q_value).pow(2)
        new_priorities = (
            new_priorities.data.cpu().numpy() + self.hyper_params["PER_EPS"]
        )
        self.memory.update_priorities(indexes, new_priorities)

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

    def write_log(self, i: int, loss: float, score: int):
        """Write log about loss and score"""
        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "epsilon: %.3f, loss: %.3f\n"
            % (i, self.episode_step, self.total_step, score, self.epsilon, loss)
        )

        if self.args.log:
            wandb.log({"score": score, "dqn loss": loss})

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            wandb.watch([self.dqn], log="parameters")

        for i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                if len(self.memory) >= self.hyper_params["UPDATE_STARTS_FROM"]:
                    experiences = self.memory.sample(self.beta)
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward

            # increase beta
            fraction = min(float(i_episode) / self.args.max_episode_steps, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # logging
            if loss_episode:
                avg_loss = np.array(loss_episode).mean(axis=0)
                self.write_log(i_episode, avg_loss, score)

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)
