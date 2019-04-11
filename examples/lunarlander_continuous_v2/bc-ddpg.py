# -*- coding: utf-8 -*-
"""Run module for BC with DDPG on LunarLanderContinuous-v2.

- Author: Curt Park, Kh Kim
- Contact: curt.park@medipixel.io
           kh.kim@medipixel.io

"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.bc.ddpg_agent import BCDDPGAgent
from algorithms.common.networks.mlp import MLP
from algorithms.common.noise import OUNoise
from examples.lunarlander_continuous_v2.utils import LunarLanderContinuousHER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 1e-3,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 512,
    "DEMO_BATCH_SIZE": 64,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "OU_NOISE_THETA": 0.0,
    "OU_NOISE_SIGMA": 0.0,
    "LAMBDA1": 1e-3,
    "LAMBDA2": 1.0,
    "WEIGHT_DECAY": 1e-6,
    "INITIAL_RANDOM_ACTION": 10000,
    "MULTIPLE_LEARN": 1,
    # HER
    "USE_HER": False,
    "SUCCESS_SCORE": 250.0,
    "DESIRED_STATES_FROM_DEMO": True,
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    if hyper_params["USE_HER"]:
        state_dim *= 2

    hidden_sizes_actor = [256, 256]
    hidden_sizes_critic = [256, 256]

    # create actor
    actor = MLP(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        output_activation=torch.tanh,
    ).to(device)

    actor_target = MLP(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        output_activation=torch.tanh,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # create critic
    critic = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic_target = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())

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

    # noise
    noise = OUNoise(
        action_dim,
        theta=hyper_params["OU_NOISE_THETA"],
        sigma=hyper_params["OU_NOISE_SIGMA"],
    )

    # make tuples to create an agent
    models = (actor, actor_target, critic, critic_target)
    optims = (actor_optim, critic_optim)

    # HER
    her = LunarLanderContinuousHER() if hyper_params["USE_HER"] else None

    # create an agent
    agent = BCDDPGAgent(env, args, hyper_params, models, optims, noise, her)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
