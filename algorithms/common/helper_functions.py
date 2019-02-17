# -*- coding: utf-8 -*-
"""Common util functions for all algorithms.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import pickle
import random
from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input without any change."""
    return x


def soft_update(local: nn.Module, target: nn.Module, tau: float):
    """Soft-update: target = tau*local + (1-tau)*target."""
    for t_param, l_param in zip(target.parameters(), local.parameters()):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


def fetch_desired_states_from_demo(demo_path: str) -> np.ndarray:
    """Return desired goal states from demonstration data."""
    with open(demo_path, "rb") as f:
        demo = pickle.load(f)

    demo = np.array(demo)
    goal_indices = np.where(demo[:, 4])[0]
    goal_states = demo[goal_indices][:, 0]

    return goal_states, goal_indices


def set_random_seed(seed: int, env: gym.Env):
    """Set random seed"""
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_env(
    env: gym.Env, args: argparse.Namespace, normalizers: List[gym.Wrapper] = None
):
    """Set environment according to user's config."""
    if args.max_episode_steps > 0:
        env._max_episode_steps = args.max_episode_steps
    else:
        args.max_episode_steps = env._max_episode_steps

    if normalizers:
        for normalizer in normalizers:
            env = normalizer(env)


def make_one_hot(labels: torch.Tensor, c: int):
    """Converts an integer label to a one-hot Variable."""
    y = torch.eye(c).to(device)
    labels = labels.type(torch.LongTensor)
    return y[labels]
