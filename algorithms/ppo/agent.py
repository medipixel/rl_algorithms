# -*- coding: utf-8 -*-
"""PPO agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

import argparse
import os
from collections import deque
from typing import Deque, Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

import algorithms.ppo.utils as ppo_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.networks.mlp import MLP, GaussianDist
from algorithms.gae import GAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "LAMBDA": 0.95,
    "EPSILON": 0.2,
    "W_VALUE": 1e-2,
    "W_ENTROPY": 1e-3,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 1e-3,
    "EPOCH": 2,
    "BATCH_SIZE": 32,
    "MIN_ROLLOUT_LEN": 128,
}


class Agent(AbstractAgent):
    """PPO Agent.

    Attributes:
        memory (deque): memory for on-policy training
        transition (list): list for storing a transition
        gae (GAE): calculator for generalized advantage estimation
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment with discrete action space
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        AbstractAgent.__init__(self, env, args)

        self.memory: Deque = deque()
        self.gae = GAE()
        self.transition: list = []

        # create models
        self.actor = GaussianDist(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=[128, 128, 128],
        ).to(device)
        self.critic = MLP(
            input_size=self.state_dim, output_size=1, hidden_sizes=[128, 128, 128]
        ).to(device)

        # create optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=hyper_params["LR_ACTOR"]
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=hyper_params["LR_CRITIC"]
        )

        # load model parameters
        if self.args.load_from is not None and os.path.exists(self.args.load_from):
            self.load_params(self.args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state_ft = torch.FloatTensor(state).to(device)
        selected_action, dist = self.actor(state_ft)

        self.transition += [
            state,
            dist.log_prob(selected_action).detach().cpu().numpy(),
        ]

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        self.transition += [action, reward, done]
        self.memory.append(self.transition)
        self.transition = []

        return next_state, reward, done

    def update_model(self) -> Tuple[float, float, float]:
        """Train the model after every N episodes."""
        states, log_probs, actions, rewards, dones = ppo_utils.decompose_memory(
            self.memory
        )

        # calculate returns and gae
        values = self.critic(states)
        returns, advantages = self.gae.get_gae(
            rewards, values, dones, hyper_params["GAMMA"], hyper_params["LAMBDA"]
        )

        actor_losses = []
        critic_losses = []
        total_losses = []
        for state, old_log_prob, action, return_, adv in ppo_utils.ppo_iter(
            hyper_params["EPOCH"],
            hyper_params["BATCH_SIZE"],
            states,
            log_probs,
            actions,
            returns,
            advantages,
        ):
            value = self.critic(state)
            _, dist = self.actor(state)

            # calculate ratios
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            surr_loss = ratio * adv
            clipped_surr_loss = (
                torch.clamp(
                    ratio, 1.0 - hyper_params["EPSILON"], 1.0 + hyper_params["EPSILON"]
                )
                * adv
            )
            actor_loss = -torch.min(surr_loss, clipped_surr_loss).mean()

            # critic_loss
            critic_loss = F.mse_loss(value, return_)

            # entropy
            entropy = dist.entropy().mean()

            # total_loss
            total_loss = (
                actor_loss
                + hyper_params["W_VALUE"] * critic_loss
                - hyper_params["W_ENTROPY"] * entropy
            )

            # train critic
            self.critic_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.data)
            critic_losses.append(critic_loss.data)
            total_losses.append(total_loss.data)

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        total_loss = sum(total_losses) / len(total_losses)

        return actor_loss, critic_loss, total_loss

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

    def write_log(self, i: int, actor_loss: float, critic_loss, total_loss, score: int):
        print(
            "[INFO] episode %d\ttotal score: %d\ttotal loss: %f\n"
            "Actor loss: %f\tCritic loss: %f\n"
            % (i, score, total_loss, actor_loss, critic_loss)
        )

        if self.args.log:
            wandb.log(
                {
                    "total loss": total_loss,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
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

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            if len(self.memory) >= hyper_params["MIN_ROLLOUT_LEN"]:
                actor_loss, critic_loss, total_loss = self.update_model()
                self.memory.clear()

                # logging
                self.write_log(i_episode, actor_loss, critic_loss, total_loss, score)

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
