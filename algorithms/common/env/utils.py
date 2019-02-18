# -*- coding: utf-8 -*-
"""Util functions for env.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
from typing import Callable, List

import gym

from algorithms.common.env.multiprocessing_env import SubprocVecEnv


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


def env_generator(
    env_name: str, args: argparse.Namespace, normalizers: List[gym.Wrapper] = None
) -> Callable:
    """Return env creating function (with normalizers)."""

    def _thunk(rank: int):
        env = gym.make(env_name)
        env.seed(args.seed + rank + 1)
        set_env(env, args, normalizers)
        return env

    return _thunk


def make_envs(env_gen: Callable, n_envs: int = 8) -> SubprocVecEnv:
    """Make multiple environments running on multiprocssors."""
    envs = [env_gen(i) for i in range(n_envs)]
    envs = SubprocVecEnv(envs)
    return envs
