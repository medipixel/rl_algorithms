# -*- coding: utf-8 -*-
"""Abstract Agent used for all agents.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from abc import ABC, abstractmethod
import argparse
import os
import subprocess
from typing import Tuple, Union

import gym
from gym.spaces import Discrete
import numpy as np
import torch
import wandb


class Agent(ABC):
    """Abstract Agent used for all agents.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        env_name (str) : gym env name for logging
        sha (str): sha code of current git commit
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
        is_discrete (bool): shows whether the action is discrete

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        self.args = args
        self.env = env

        if isinstance(env.action_space, Discrete):
            self.is_discrete = True
        else:
            self.is_discrete = False

        # for logging
        self.env_name = str(self.env.env).split("<")[2].replace(">>", "")
        self.sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])[:-1]
            .decode("ascii")
            .strip()
        )

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abstractmethod
    def step(
        self, action: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def update_model(self, *args) -> Tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def load_params(self, *args):
        pass

    @abstractmethod
    def save_params(self, params: dict, n_episode: int):
        if not os.path.exists("./save"):
            os.mkdir("./save")

        save_name = self.env_name + "_" + self.args.algo + "_" + self.sha

        path = os.path.join("./save/" + save_name + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)

        print("[INFO] Saved the model and optimizer to", path)

    @abstractmethod
    def write_log(self, *args):
        pass

    @abstractmethod
    def train(self):
        pass

    def interim_test(self):
        self.args.test = True

        print()
        print("===========")
        print("Start Test!")
        print("===========")

        self._test(interim_test=True)

        print("===========")
        print("Test done!")
        print("===========")
        print()

        self.args.test = False

    def test(self):
        """Test the agent."""
        # logger
        if self.args.log:
            wandb.init()

        self._test()

        # termination
        self.env.close()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.args.interim_test_num
        else:
            test_num = self.args.episode_num

        for i_episode in range(test_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.args.render:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )

            if self.args.log:
                wandb.log({"test score": score})
