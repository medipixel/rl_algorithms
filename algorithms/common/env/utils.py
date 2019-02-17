# -*- coding: utf-8 -*-
"""Util functions for env.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
from typing import Callable, List

import gym

import algorithms.common.helper_functions as common_utils
from algorithms.common.multiprocessing_env import SubprocVecEnv


def env_generator(
    env_name: str, args: argparse.Namespace, normalizers: List[gym.Wrapper] = None
) -> Callable:
    """Return env creating function (with normalizers)."""

    def _thunk(rank: int):
        env = gym.make(env_name)
        env.seed(args.seed + rank + 1)
        common_utils.set_env(env, args, normalizers)
        return env

    return _thunk


def make_envs(env_gen: Callable, n_envs: int = 8) -> SubprocVecEnv:
    """Make multiple environments running on multiprocssors."""
    envs = [env_gen(i) for i in range(n_envs)]
    envs = SubprocVecEnv(envs)
    return envs
