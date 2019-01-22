# -*- coding: utf-8 -*-
"""Abstract Agent used for all agents.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import os
from abc import ABC, abstractmethod

import git
import torch

from torch_env import TorchEnv


class AbstractAgent(ABC):
    """Abstract Agent used for all agents.

    Args:
        env (TorchEnv): openAI Gym environment with discrete action space
        args (argparse.Namespace): arguments including hyperparameters and training settings

    Attributes:
        env (gym.Env): openAI Gym environment with discrete action space
        args (argparse.Namespace): arguments including hyperparameters and training settings
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of the action value
        action_high (float): upper bound of the action value

    """

    def __init__(self, env: TorchEnv, args: argparse.Namespace):
        """Initialization."""
        self.args = args
        self.env = env
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.action_low = self.env.action_low
        self.action_high = self.env.action_high

    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def update_model(self):
        pass

    @abstractmethod
    def load_params(self):
        pass

    @abstractmethod
    def save_params(self, name: str, params: dict, n_episode: int):
        if not os.path.exists("./save"):
            os.mkdir("./save")

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        path = os.path.join(
            "./save/" + name + "_" + sha[:7] + "_ep_" + str(n_episode) + ".pt"
        )
        torch.save(params, path)

        print("[INFO] Saved the model and optimizer to", path)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
