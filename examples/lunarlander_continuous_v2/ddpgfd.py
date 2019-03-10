# -*- coding: utf-8 -*-
"""Run module for DDPGfD on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP
from algorithms.common.noise import OUNoise
from algorithms.fd.ddpg_agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "N_STEP": 1,
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 64,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 3e-4,
    "OU_NOISE_THETA": 0.0,
    "OU_NOISE_SIGMA": 0.0,
    "PRETRAIN_STEP": 0,
    "MULTIPLE_LEARN": 2,  # multiple learning updates
    "LAMBDA1": 1.0,  # N-step return weight
    "LAMBDA2": 1e-5,  # l2 regularization weight
    "LAMBDA3": 1.0,  # actor loss contribution of prior weight
    "PER_ALPHA": 0.3,
    "PER_BETA": 1.0,
    "PER_EPS": 1e-6,
    "PER_EPS_DEMO": 1.0,
    "INITIAL_RANDOM_ACTION": int(5e3),
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
        weight_decay=hyper_params["LAMBDA2"],
    )

    critic_optim = optim.Adam(
        critic.parameters(),
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["LAMBDA2"],
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

    # create an agent
    agent = Agent(env, args, hyper_params, models, optims, noise)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
