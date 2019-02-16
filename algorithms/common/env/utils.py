# -*- coding: utf-8 -*-
"""Util functions for env.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Callable, List

import gym

from algorithms.common.multiprocessing_env import SubprocVecEnv


def env_generator(env_name: str, normalizers: List[gym.Wrapper] = None) -> Callable:
    """return env creating function (with normalizers)."""

    def _thunk():
        env = gym.make(env_name)
        if normalizers:
            for normalizer in normalizers:
                env = normalizer(env)
        return env

    return _thunk


def make_envs(env_gen: Callable, n_envs: int = 8) -> SubprocVecEnv:
    """Make multiple environments running on multiprocssors."""
    envs = [env_gen() for _ in range(n_envs)]
    envs = SubprocVecEnv(envs)
    return envs
