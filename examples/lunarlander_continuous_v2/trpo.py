# -*- coding: utf-8 -*-
"""Run module for TRPO on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch

from algorithms.common.networks.mlp import MLP, GaussianDistParams
from algorithms.trpo.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.95,
    "LAMBDA": 0.9,
    "MAX_KL": 1e-2,
    "DAMPING": 1e-1,
    "L2_REG": 1e-3,
    "LBFGS_MAX_ITER": 128,
    "MIN_ROLLOUT_LEN": 64,
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    hidden_sizes_actor = [256, 256]
    hidden_sizes_critic = [256, 256]

    # create models
    actor = GaussianDistParams(
        input_size=state_dim, output_size=action_dim, hidden_sizes=hidden_sizes_actor
    ).to(device)

    old_actor = GaussianDistParams(
        input_size=state_dim, output_size=action_dim, hidden_sizes=hidden_sizes_actor
    ).to(device)

    critic = MLP(
        input_size=state_dim, output_size=1, hidden_sizes=hidden_sizes_critic
    ).to(device)

    # make tuples to create an agent
    models = (actor, old_actor, critic)

    # create an agent
    agent = Agent(env, args, hyper_params, models)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
