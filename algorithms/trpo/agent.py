# -*- coding: utf-8 -*-
"""TRPO agent for episodic tasks in OpenAI Gym.

The overall implementation is very inspired by
https://github.com/ikostrikov/pytorch-trpo

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://arxiv.org/abs/1502.05477

"""

import argparse
import os
from collections import deque
from typing import Deque, Tuple

import gym
import numpy as np
import torch
import wandb

import algorithms.trpo.utils as trpo_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.gae import GAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """TRPO Agent.

    Attributes:
        memory (deque): memory for on-policy training
        gae (GAE): calculator for generalized advantage estimation
        actor (nn.Module): policy gradient model to select actions
        old_actor (nn.Module): old policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values
        hyper_params (dict): hyper-parameters
        curr_state (np.ndarray): temporary storage of the current state

    """

    def __init__(
        self, env: gym.Env, args: argparse.Namespace, hyper_params: dict, models: tuple
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
        """

        AbstractAgent.__init__(self, env, args)

        self.actor, self.old_actor, self.critic = models
        self.hyper_params = hyper_params
        self.memory: Deque = deque()
        self.gae = GAE()
        self.curr_state = np.zeros((1,))

        # load model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        self.curr_state = state

        state = torch.FloatTensor(state).to(device)
        mu, _, std = self.actor(state)
        selected_action = torch.normal(mu, std)

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        self.memory.append([self.curr_state, action, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after every N episodes."""
        self.old_actor.load_state_dict(self.actor.state_dict())

        states, actions, rewards, dones = trpo_utils.decompose_memory(self.memory)

        # calculate returns and gae
        values = self.critic(states)
        returns, advantages = self.gae.get_gae(
            rewards,
            values,
            dones,
            self.hyper_params["GAMMA"],
            self.hyper_params["LAMBDA"],
        )

        # train critic
        critic_loss = trpo_utils.critic_step(
            self.critic,
            states,
            returns.detach(),
            self.hyper_params["L2_REG"],
            self.hyper_params["LBFGS_MAX_ITER"],
        )

        # train actor
        actor_loss = trpo_utils.actor_step(
            self.old_actor,
            self.actor,
            states,
            actions,
            advantages,
            self.hyper_params["MAX_KL"],
            self.hyper_params["DAMPING"],
        )

        return actor_loss.data, critic_loss.data

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
        }

        AbstractAgent.save_params(self, params, n_episode)

    def write_log(self, i: int, loss: np.ndarray, score: int):
        """Write log about loss and score"""
        total_loss = loss.sum()

        print(
            "[INFO] episode %d total score: %d, total loss: %f\n"
            "actor_loss: %.3f critic_loss: %.3f\n"
            % (i, score, total_loss, loss[0], loss[1])  # actor loss  # critic loss
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0],
                    "critic loss": loss[1],
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            wandb.watch([self.actor, self.critic], log="parameters")

        for i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            if len(self.memory) >= self.hyper_params["MIN_ROLLOUT_LEN"]:
                loss = self.update_model()
                loss_episode.append(loss)  # for logging
                self.memory.clear()

            # for logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(i_episode, avg_loss, score)

                if i_episode % self.args.save_period == 0:
                    self.save_params(i_episode)

        # termination
        self.env.close()
