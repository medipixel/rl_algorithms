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
import scipy.optimize
import torch
import wandb

import algorithms.trpo.utils as trpo_utils
from algorithms.abstract_agent import AbstractAgent
from algorithms.gae import GAE
from algorithms.trpo.model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.95,
    "LAMBDA": 0.9,
    "MAX_KL": 1e-2,
    "DAMPING": 1e-1,
    "L2_REG": 1e-3,
    "LBFGS_MAX_ITER": 200,
    "BATCH_SIZE": 256,
    "MAX_EPISODE_STEPS": 300,
    "EPISODE_NUM": 1500,
}


class Agent(AbstractAgent):
    """TRPO Agent.

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        args (argparse.Namespace): arguments including hyperparameters and training settings

    Attributes:
        memory (deque): memory for on-policy training
        gae (GAE): calculator for generalized advantage estimation
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """Initialization."""
        AbstractAgent.__init__(self, env, args)

        self.memory: Deque = deque()
        self.get_gae = GAE()

        # environment setup
        self.env._max_episode_steps = hyper_params["MAX_EPISODE_STEPS"]

        # create models
        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_low, self.action_high
        ).to(device)
        self.old_actor = Actor(
            self.state_dim, self.action_dim, self.action_low, self.action_high
        ).to(device)
        self.critic = Critic(self.state_dim, self.action_dim).to(device)

        # load model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = torch.FloatTensor(state).to(device)
        selected_action, dist = self.actor(state)

        return selected_action

    def step(
        self, state: np.ndarray, action: torch.Tensor
    ) -> Tuple[np.ndarray, np.float64, bool]:
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        self.memory.append([state, action, reward, done])

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Train the model after every N episodes."""
        self.old_actor.load_state_dict(self.actor.state_dict())

        states, actions, rewards, dones = trpo_utils.decompose_memory(self.memory)

        # calculate returns and gae
        values = self.critic(states)
        returns, advantages = self.get_gae(
            rewards, values, dones, hyper_params["GAMMA"], hyper_params["LAMBDA"]
        )

        # train critic
        targets = returns.detach()
        get_value_loss = trpo_utils.ValueLoss(
            self.critic, states, targets, hyper_params["L2_REG"]
        )
        flat_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
            get_value_loss,
            trpo_utils.get_flat_params_from(self.critic).cpu().double().numpy(),
            maxiter=hyper_params["LBFGS_MAX_ITER"],
        )
        trpo_utils.set_flat_params_to(self.critic, torch.Tensor(flat_params))
        critic_loss = (self.critic(states) - targets).pow(2).mean()

        # train actor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        actor_loss = trpo_utils.trpo_step(
            self.old_actor,
            self.actor,
            states,
            actions,
            advantages,
            hyper_params["MAX_KL"],
            hyper_params["DAMPING"],
        )

        # for logging
        total_loss = actor_loss + critic_loss

        return total_loss.data

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

        AbstractAgent.save_params(self, "trpo", params, n_episode)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor, self.critic], log="parameters")

        loss = 0
        for i_episode in range(1, hyper_params["EPISODE_NUM"] + 1):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(state, action)

                state = next_state
                score += reward

                if len(self.memory) % hyper_params["BATCH_SIZE"] == 0:
                    loss = self.update_model()
                    self.memory.clear()

            else:
                print(
                    "[INFO] episode %d\ttotal score: %d\trecent loss: %f"
                    % (i_episode, score, loss)
                )

                if self.args.log:
                    wandb.log({"recent loss": loss, "score": score})

                if i_episode % self.args.save_period == 0:
                    self.save_params(i_episode)

        # termination
        self.env.close()
        self.save_params(hyper_params["EPISODE_NUM"])

    def test(self):
        """Test the agent."""
        for i_episode in range(hyper_params["EPISODE_NUM"]):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(state, action)

                state = next_state
                score += reward

            else:
                print("[INFO] episode %d\ttotal score: %d" % (i_episode, score))

        # termination
        self.env.close()
