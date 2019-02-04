# -*- coding: utf-8 -*-
"""Reinforce agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
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

from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.networks.mlp import MLP, GaussianDist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {"GAMMA": 0.99, "LR_ACTOR": 1e-3, "LR_BASELINE": 1e-3}


class Agent(AbstractAgent):
    """ReinforceAgent interacting with environment.

    Attributes:
        actor (nn.Module): policy model to select actions
        critic (nn.Module): critic model to evaluate states
        actor_optimizer (Optimizer): optimizer for actor
        critic_optimizer (Optimizer): optimizer for critic
        log_prob_sequence (list): log probabailities of an episode
        predicted_value_sequence (list): predicted values of an episode
        reward_sequence (list): rewards of an episode to calculate returns

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment with discrete action space
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        AbstractAgent.__init__(self, env, args)

        self.log_prob_sequence: list = []
        self.predicted_value_sequence: list = []
        self.reward_sequence: list = []

        # create models
        self.actor = GaussianDist(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=[256, 256],
        ).to(device)

        self.baseline = MLP(
            input_size=self.state_dim, output_size=1, hidden_sizes=[256, 256]
        ).to(device)

        # create optimizer
        lr_actor, lr_baseline = hyper_params["LR_ACTOR"], hyper_params["LR_BASELINE"]
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr_actor)
        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=lr_baseline)

        # load stored parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = torch.FloatTensor(state).to(device)

        selected_action, dist = self.actor(state)
        predicted_value = self.baseline(state)

        self.log_prob_sequence.append(dist.log_prob(selected_action).sum(dim=-1))
        self.predicted_value_sequence.append(predicted_value)

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        # store rewards to calculate return values
        self.reward_sequence.append(reward)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        return_value = 0  # initial return value
        return_sequence: Deque = deque()

        # calculate return value at each step
        for i in range(len(self.reward_sequence) - 1, -1, -1):
            return_value = (
                self.reward_sequence[i] + hyper_params["GAMMA"] * return_value
            )
            return_sequence.appendleft(return_value)

        # standardize returns for better stability
        return_sequence_tensor = torch.Tensor(return_sequence).to(device)
        return_sequence_tensor = (
            return_sequence_tensor - return_sequence_tensor.mean()
        ) / (return_sequence_tensor.std() + 1e-7)

        # calculate loss at each step
        policy_loss_sequence = []
        value_loss_sequence = []
        for log_prob, return_value, predicted_value in zip(
            self.log_prob_sequence,
            return_sequence_tensor,
            self.predicted_value_sequence,
        ):
            delta = return_value - predicted_value.detach()

            policy_loss = -delta * log_prob
            value_loss = F.mse_loss(predicted_value, return_value)

            policy_loss_sequence.append(policy_loss)
            value_loss_sequence.append(value_loss)

        # train
        self.actor_optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss_sequence).mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.baseline_optimizer.zero_grad()
        value_loss = torch.stack(value_loss_sequence).mean()
        value_loss.backward()
        self.baseline_optimizer.step()

        # clear
        self.log_prob_sequence.clear()
        self.predicted_value_sequence.clear()
        self.reward_sequence.clear()

        return policy_loss.data, value_loss.data

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.baseline.load_state_dict(params["baseline_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.baseline_optimizer.load_state_dict(params["baseline_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "baseline_state_dict": self.baseline.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "baseline_optim_state_dict": self.baseline_optimizer.state_dict(),
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
        """Run the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor, self.baseline], log="parameters")

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

            policy_loss, value_loss = self.update_model()

            # logging
            self.write_log(i_episode, score, policy_loss, value_loss)

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
