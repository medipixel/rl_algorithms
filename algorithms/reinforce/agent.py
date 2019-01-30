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
from algorithms.reinforce.model import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {"GAMMA": 0.99, "STD": 1.0, "LR_MODEL": 1e-3}


class Agent(AbstractAgent):
    """ReinforceAgent interacting with environment.

    Attributes:
        model (nn.Module): policy gradient model to select actions
        optimizer (Optimizer): optimizer for training
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

        # create a model
        self.model = ActorCritic(
            hyper_params["STD"], self.state_dim, self.action_dim
        ).to(device)

        # create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=hyper_params["LR_MODEL"]
        )

        # load stored parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = torch.FloatTensor(state).to(device)
        selected_action, predicted_value, dist = self.model(state)

        self.log_prob_sequence.append(dist.log_prob(selected_action).sum())
        self.predicted_value_sequence.append(predicted_value)

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        # store rewards to calculate return values
        self.reward_sequence.append(reward)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
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
        loss_sequence = []
        for log_prob, return_value, predicted_value in zip(
            self.log_prob_sequence,
            return_sequence_tensor,
            self.predicted_value_sequence,
        ):
            delta = return_value - predicted_value.detach()

            policy_loss = -delta * log_prob
            value_loss = F.smooth_l1_loss(predicted_value, return_value)

            loss = policy_loss + value_loss
            loss_sequence.append(loss)

        # train
        self.optimizer.zero_grad()
        total_loss = torch.stack(loss_sequence).sum()
        total_loss.backward()
        self.optimizer.step()

        # clear
        self.log_prob_sequence.clear()
        self.predicted_value_sequence.clear()
        self.reward_sequence.clear()

        return total_loss

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.model.load_state_dict(params["model_state_dict"])
        self.optimizer.load_state_dict(params["optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
        }

        AbstractAgent.save_params(self, self.args.algo, params, n_episode)

    def train(self):
        """Run the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch(self.model, log="parameters")

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

            loss = self.update_model()

            # logging
            print(
                "[INFO] episode %d\ttotal score: %d\tloss: %f"
                % (i_episode, score, loss)
            )

            if self.args.log:
                wandb.log({"score": score, "loss": loss})

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
