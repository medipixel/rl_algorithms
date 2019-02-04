# -*- coding: utf-8 -*-
"""DPG agent for episodic tasks in OpenAI Gym.

This algorithm doesn't guarantee convergence becasue
the model structures don't complies theorem 4.3 in the paper.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://proceedings.mlr.press/v32/silver14.pdf
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from algorithms.common.abstract.agent import AbstractAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """ActorCritic interacting with environment.

    Attributes:
        actor (nn.Module): actor model to select actions
        critic (nn.Module): critic model to predict values
        hyper_params (dict): hyper-parameters
        actor_optimizer (Optimizer): actor optimizer for training
        critic_optimizer (Optimizer): critic optimizer for training

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic

        """
        AbstractAgent.__init__(self, env, args)

        self.actor, self.critic = models
        self.actor_optimizer, self.critic_optimizer = optims
        self.hyper_params = hyper_params

        # load stored parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state)

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done

    def update_model(
        self, experience: Tuple[np.ndarray, torch.Tensor, np.float64, np.ndarray, bool]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        state, action, reward, next_state, done = experience
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        mask = 1 - done

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        value = self.critic(torch.cat((state, action), dim=-1))
        next_action = self.actor(next_state)
        next_value = self.critic(torch.cat((next_state, next_action), dim=-1)).detach()
        curr_return = reward + self.hyper_params["GAMMA"] * next_value * mask
        curr_return = curr_return.to(device)

        # train critic
        critic_loss = F.mse_loss(value, curr_return)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        action = self.actor(state)
        actor_loss = -self.critic(torch.cat((state, action), dim=-1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.data, critic_loss.data

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }

        AbstractAgent.save_params(self, self.args.algo, params, n_episode)

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

                loss = self.update_model((state, action, reward, next_state, done))

                state = next_state
                score += reward

                loss_episode.append(loss)  # for logging

            # logging
            avg_loss = np.vstack(loss_episode).mean(axis=0)
            self.write_log(i_episode, avg_loss, score)

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
