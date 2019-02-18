# -*- coding: utf-8 -*-
"""Run module for PPO on LunarLander-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.env.utils import env_generator, make_envs
from algorithms.common.networks.mlp import MLP, CategoricalDist
from algorithms.ppo.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "LAMBDA": 0.95,
    "EPSILON": 0.2,
    "MIN_EPSILON": 0.1,
    "EPSILON_DECAY_PERIOD": 1500,
    "W_VALUE": 0.5,
    "W_ENTROPY": 1e-3,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 1e-3,
    "EPOCH": 10,
    "BATCH_SIZE": 64,
    "ROLLOUT_LEN": 1024,
    "GRADIENT_CLIP": 0.5,
    "WEIGHT_DECAY": 0,
    "N_WORKERS": 16,
    "N_TEST": 3,
    "USE_CLIPPED_VALUE_LOSS": True,
    "STANDARDIZE_ADVANTAGE": True,
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # create multiple envs
    env_test = env
    env_gen = env_generator("LunarLander-v2", args)
    envs_train = make_envs(env_gen, n_envs=hyper_params["N_WORKERS"])

    # create models
    hidden_sizes_actor = [128, 128, 64]
    hidden_sizes_critic = [128, 128, 64]

    actor = CategoricalDist(
        input_size=state_dim, output_size=action_dim, hidden_sizes=hidden_sizes_actor
    ).to(device)

    critic = MLP(
        input_size=state_dim, output_size=1, hidden_sizes=hidden_sizes_critic
    ).to(device)

    # create optimizer
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    critic_optim = optim.Adam(
        critic.parameters(),
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    # make tuples to create an agent
    models = (actor, critic)
    optims = (actor_optim, critic_optim)

    # create an agent
    agent = Agent(env_test, envs_train, args, hyper_params, models, optims)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
