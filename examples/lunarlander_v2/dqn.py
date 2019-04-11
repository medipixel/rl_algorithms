# -*- coding: utf-8 -*-
"""Run module for DQN on LunarLander-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import argparse

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common.helper_functions import identity
from algorithms.common.networks.mlp import init_layer_uniform
from algorithms.dqn.agent import DQNAgent
from algorithms.dqn.linear import NoisyLinearConstructor
from algorithms.dqn.networks import C51DuelingMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "N_STEP": 3,
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "W_N_STEP": 1.0,
    "W_Q_REG": 1e-7,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 64,
    "LR_DQN": 1e-4,  # dueling: 6.25e-5
    "ADAM_EPS": 1e-8,  # rainbow: 1.5e-4
    "WEIGHT_DECAY": 1e-7,
    "MAX_EPSILON": 1.0,
    "MIN_EPSILON": 0.01,
    "EPSILON_DECAY": 1e-5,
    "PER_ALPHA": 0.6,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "GRADIENT_CLIP": 10.0,
    "UPDATE_STARTS_FROM": int(1e4),
    "TRAIN_FREQ": 1,
    "MULTIPLE_LEARN": 1,
    # Distributional Q function
    "USE_DIST_Q": "C51",
    "V_MIN": -300,
    "V_MAX": 300,
    "ATOMS": 1530,
    # NoisyNet
    "USE_NOISY_NET": True,
    "STD_INIT": 0.5,
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # create model
    def get_fc_model():
        hidden_sizes = [128, 64]

        if hyper_params["USE_NOISY_NET"]:
            # use noisy net
            linear_layer = NoisyLinearConstructor(hyper_params["STD_INIT"])
            init_fn = identity
            hyper_params["MAX_EPSILON"] = 0.0
            hyper_params["MIN_EPSILON"] = 0.0
        else:
            linear_layer = nn.Linear
            init_fn = init_layer_uniform

        model = C51DuelingMLP(
            input_size=state_dim,
            action_size=action_dim,
            hidden_sizes=hidden_sizes,
            v_min=hyper_params["V_MIN"],
            v_max=hyper_params["V_MAX"],
            atom_size=hyper_params["ATOMS"],
            linear_layer=linear_layer,
            init_fn=init_fn,
        ).to(device)

        return model

    dqn = get_fc_model()
    dqn_target = get_fc_model()
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
    agent = DQNAgent(env, args, hyper_params, models, dqn_optim)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
