# -*- coding: utf-8 -*-
"""Run module for PPO on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import multiprocessing

import gym
import torch
import torch.optim as optim

from algorithms.common.env.utils import env_generator, make_envs
from algorithms.common.networks.mlp import MLP, GaussianDist
from algorithms.ppo.agent import PPOAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "LAMBDA": 0.95,
    "EPSILON": 0.2,
    "MIN_EPSILON": 0.2,
    "EPSILON_DECAY_PERIOD": 1500,
    "W_VALUE": 0.5,
    "W_ENTROPY": 1e-3,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 1e-3,
    "EPOCH": 16,
    "BATCH_SIZE": 32,
    "ROLLOUT_LEN": 256,
    "GRADIENT_CLIP_AC": 0.5,
    "GRADIENT_CLIP_CR": 0.5,
    "WEIGHT_DECAY": 0,
    "N_WORKERS": multiprocessing.cpu_count(),
    "USE_CLIPPED_VALUE_LOSS": True,
    "STANDARDIZE_ADVANTAGE": True,
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
    env_gen = env_generator("LunarLanderContinuous-v2", args)
    env_multi = make_envs(env_gen, n_envs=hyper_params["N_WORKERS"])

    # create models
    hidden_sizes_actor = [256, 256]
    hidden_sizes_critic = [256, 256]

    actor = GaussianDist(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        hidden_activation=torch.tanh,
    ).to(device)

    critic = MLP(
        input_size=state_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
        hidden_activation=torch.tanh,
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
    agent = PPOAgent(env_single, env_multi, args, hyper_params, models, optims)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
