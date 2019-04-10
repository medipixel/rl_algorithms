# -*- coding: utf-8 -*-
"""Run module for TD3 on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP
from algorithms.common.noise import GaussianNoise
from algorithms.td3.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(1e6),
    "BATCH_SIZE": 100,
    "LR_ACTOR": 1e-3,
    "LR_CRITIC": 1e-3,
    "DELAYED_UPDATE": 2,
    "TARGET_SMOOTHING_NOISE_STD": 0.2,
    "TARGET_SMOOTHING_NOISE_CLIP": 0.5,
    "GAUSSIAN_NOISE_MIN_SIGMA": 0.1,
    "GAUSSIAN_NOISE_MAX_SIGMA": 0.1,
    "GAUSSIAN_NOISE_DECAY_PERIOD": int(1e6),
    "WEIGHT_DECAY": 0.0,
    "INITIAL_RANDOM_ACTION": int(1e4),
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    hidden_sizes_actor = [400, 300]
    hidden_sizes_critic = [400, 300]

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
    critic_1 = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic_2 = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic_target1 = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic_target2 = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic_target1.load_state_dict(critic_1.state_dict())
    critic_target2.load_state_dict(critic_2.state_dict())

    # create optimizers
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    critic_parameter = list(critic_1.parameters()) + list(critic_2.parameters())
    critic_optim = optim.Adam(
        critic_parameter,
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    # noise instance to make randomness of action
    noise = GaussianNoise(
        hyper_params["GAUSSIAN_NOISE_MIN_SIGMA"],
        hyper_params["GAUSSIAN_NOISE_MAX_SIGMA"],
        hyper_params["GAUSSIAN_NOISE_DECAY_PERIOD"],
    )

    # make tuples to create an agent
    models = (actor, actor_target, critic_1, critic_2, critic_target1, critic_target2)
    optims = (actor_optim, critic_optim)

    # create an agent
    agent = Agent(env, args, hyper_params, models, optims, noise)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
