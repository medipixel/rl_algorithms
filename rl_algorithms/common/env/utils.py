# -*- coding: utf-8 -*-
"""Util functions for env.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Callable, List, Tuple

import gym
from gym.spaces import Discrete

from rl_algorithms.common.env.multiprocessing_env import SubprocVecEnv
from rl_algorithms.common.env.normalizers import ActionNormalizer


def set_env(
    env: gym.Env, max_episode_steps: int, env_wrappers: List[gym.Wrapper] = None
) -> Tuple[gym.Env, int]:
    """Set environment according to user's config."""
    if max_episode_steps > 0:
        env._max_episode_steps = max_episode_steps
    else:
        max_episode_steps = env._max_episode_steps

    if not isinstance(env.action_space, Discrete):
        env = ActionNormalizer(env)

    if env_wrappers:
        for env_wrapper in env_wrappers:
            env = env_wrapper(env)

    return env, max_episode_steps


def env_generator(
    env_name: str, max_episode_steps: int, env_wrappers: List[gym.Wrapper] = None
) -> Callable:
    """Return env creating function (with normalizers)."""

    def _thunk(rank: int):
        env = gym.make(env_name)
        env.seed(777 + rank + 1)
        env, _ = set_env(env, max_episode_steps, env_wrappers)
        return env

    return _thunk


def make_envs(env_gen: Callable, n_envs: int = 8) -> SubprocVecEnv:
    """Make multiple environments running on multiprocssors."""
    envs = [env_gen(i) for i in range(n_envs)]
    subproc_env = SubprocVecEnv(envs)
    return subproc_env
