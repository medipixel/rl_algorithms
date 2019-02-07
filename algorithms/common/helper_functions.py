# -*- coding: utf-8 -*-
"""Common util functions for all algorithms.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import math
import pickle
import random

import gym
import numpy as np
import torch
import torch.nn as nn


# taken and modified from https://github.com/ikostrikov/pytorch-trpo
def normal_log_density(
    x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor, std: torch.Tensor
):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(-1, keepdim=True)


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
