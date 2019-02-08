# -*- coding: utf-8 -*-
"""Abstract Agent used for all agents.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Tuple

import gym
import numpy as np
import torch


class AbstractAgent(ABC):
    """Abstract Agent used for all agents.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        env_name (str) : gym env name for logging
        sha (str): sha code of current git commit

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        self.args = args
        self.env = NormalizedActions(env)
        if self.args.max_episode_steps > 0:
            env._max_episode_steps = self.args.max_episode_steps
        else:
            self.args.max_episode_steps = env._max_episode_steps

        # for logging
        self.env_name = str(self.env.env).split("<")[2].replace(">>", "")
        self.sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])[:-1]
            .decode("ascii")
            .strip()
        )

    @abstractmethod
    def select_action(self, state: np.ndarray):
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        pass

    @abstractmethod
    def update_model(self, *args):
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

    def test(self):
        """Test the agent."""
        for i_episode in range(self.args.episode_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
                step += 1

            print(
                "[INFO] episode %d\tstep: %d\ttotal score: %d"
                % (i_episode, step, score)
            )

        # termination
        self.env.close()


class NormalizedActions(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action
