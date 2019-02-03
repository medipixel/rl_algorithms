# -*- coding: utf-8 -*-
"""1-Step Advantage Actor-Critic agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.networks.mlp import MLP, GaussianDist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# hyper parameters
hyper_params = {"GAMMA": 0.99, "LR_ACTOR": 1e-4, "LR_CRITIC": 1e-3, "WEIGHT_DECAY": 0.0}


class Agent(AbstractAgent):
    """1-Step Advantage Actor-Critic interacting with environment.

    Attributes:
        actor (nn.Module): policy model to select actions
        critic (nn.Module): critic model to evaluate states
        actor_optimizer (Optimizer): optimizer for actor
        critic_optimizer (Optimizer): optimizer for critic
        optimizer (Optimizer): optimizer for training

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment with discrete action space
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        AbstractAgent.__init__(self, env, args)

        self.log_prob = torch.zeros((1,))
        self.predicted_value = torch.zeros((1,))

        # create models
        self.actor = GaussianDist(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=[48, 48],
        ).to(device)

        self.critic = MLP(
            input_size=self.state_dim, output_size=1, hidden_sizes=[48, 48]
        ).to(device)

        # create optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=hyper_params["LR_ACTOR"],
            weight_decay=hyper_params["WEIGHT_DECAY"],
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=hyper_params["LR_CRITIC"],
            weight_decay=hyper_params["WEIGHT_DECAY"],
        )

        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = torch.FloatTensor(state).to(device)

        selected_action, dist = self.actor(state)
        predicted_value = self.critic(state)

        self.log_prob = dist.log_prob(selected_action).sum(dim=-1)
        self.predicted_value = predicted_value

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done

    def update_model(
        self, experience: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reward, next_state, done = experience
        next_state = torch.FloatTensor(next_state).to(device)

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        next_value = self.critic(next_state).detach()
        q_value = reward + hyper_params["GAMMA"] * next_value * mask
        q_value = q_value.to(device)

        # advantage = Q_t - V(s_t)
        advantage = q_value - self.predicted_value

        # calculate loss at the current step
        policy_loss = -advantage.detach() * self.log_prob  # adv. is not backpropagated
        value_loss = F.mse_loss(self.predicted_value, q_value.detach())

        # train
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        return policy_loss.data, value_loss.data

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[INFO] The input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] Loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }

        AbstractAgent.save_params(self, self.args.algo, params, n_episode)

    def write_log(self, i: int, score: int, policy_loss: float, value_loss: float):
        total_loss = policy_loss + value_loss

        print(
            "[INFO] episode %d\ttotal score: %d\ttotal loss: %f\n"
            "policy loss: %f\tvalue loss: %f\n"
            % (i, score, total_loss, policy_loss, value_loss)
        )

        if self.args.log:
            wandb.log(
                {
                    "total loss": total_loss,
                    "policy loss": policy_loss,
                    "value loss": value_loss,
                    "score": score,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor, self.critic], log="parameters")

        for i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            policy_loss_episode = list()
            value_loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                policy_loss, value_loss = self.update_model((reward, next_state, done))
                policy_loss_episode.append(policy_loss)
                value_loss_episode.append(value_loss)

                state = next_state
                score += reward

            # logging
            policy_loss = np.array(policy_loss_episode).mean()
            value_loss = np.array(value_loss_episode).mean()
            self.write_log(i_episode, score, policy_loss, value_loss)

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
