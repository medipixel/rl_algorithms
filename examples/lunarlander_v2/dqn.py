# -*- coding: utf-8 -*-
"""Run module for DQN on LunarLander-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.env.utils import env_generator, make_envs
from algorithms.common.networks.mlp import MLP
from algorithms.dqn.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.999,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 32,
    "LR_DQN": 1e-5,
    "WEIGHT_DECAY": 1e-6,
    "MAX_EPSILON": 1.0,
    "MIN_EPSILON": 0.01,
    "EPSILON_DECAY": 1e-5,
    "PER_ALPHA": 0.5,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "UPDATE_STARTS_FROM": int(1e4),
    "N_WORKERS": 16,
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # create multiple envs
    env_single = env
    env_gen = env_generator("LunarLander-v2", args)
    env_multi = make_envs(env_gen, n_envs=hyper_params["N_WORKERS"])

    # create model
    hidden_sizes = [128, 64]

    dqn = MLP(
        input_size=state_dim, output_size=action_dim, hidden_sizes=hidden_sizes
    ).to(device)

    dqn_target = MLP(
        input_size=state_dim, output_size=action_dim, hidden_sizes=hidden_sizes
    ).to(device)
    dqn_target.load_state_dict(dqn.state_dict())

    # create optimizer
    dqn_optim = optim.Adam(
        dqn.parameters(),
        lr=hyper_params["LR_DQN"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    # make tuples to create an agent
    models = (dqn, dqn_target)

    # create an agent
    agent = Agent(env_single, env_multi, args, hyper_params, models, dqn_optim)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
