# -*- coding: utf-8 -*-
"""Run module for DQN on LunarLander-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.dqn.networks import CategoricalDuelingMLP, DuelingMLP
from algorithms.fd.dqn_agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "N_STEP": 3,
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 64,
    "LR_DQN": 1e-4,  # dueling: 6.25e-5
    "ADAM_EPS": 1e-8,  # rainbow: 1.5e-4
    "MAX_EPSILON": 1.0,
    "MIN_EPSILON": 0.01,
    "EPSILON_DECAY": 2e-5,
    "PER_ALPHA": 0.6,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-3,
    "PER_EPS_DEMO": 1.0,
    "PRETRAIN_STEP": int(1e2),
    "LAMBDA1": 1.0,  # N-step return weight
    "LAMBDA2": 1.0,  # Supervised loss weight
    "LAMBDA3": 1e-5,  # l2 regularization weight
    "W_Q_REG": 1e-7,  # Q value regularization
    "MARGIN": 0.8,  # margin for supervised loss
    "GRADIENT_CLIP": 0.5,
    "UPDATE_STARTS_FROM": int(1e3),
    "TRAIN_FREQ": 8,
    "MULTIPLE_LEARN": 1,
    # C51
    "USE_C51": False,
    "V_MIN": -10,
    "V_MAX": 10,
    "ATOMS": 51,
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

        if hyper_params["USE_C51"]:
            model = CategoricalDuelingMLP(
                input_size=state_dim,
                action_size=action_dim,
                hidden_sizes=hidden_sizes,
                v_min=hyper_params["V_MIN"],
                v_max=hyper_params["V_MAX"],
                atom_size=hyper_params["ATOMS"],
            ).to(device)

        else:
            model = DuelingMLP(
                input_size=state_dim, output_size=action_dim, hidden_sizes=hidden_sizes
            ).to(device)

        return model

    dqn = get_fc_model()
    dqn_target = get_fc_model()
    dqn_target.load_state_dict(dqn.state_dict())

    # create optimizer
    dqn_optim = optim.Adam(
        dqn.parameters(),
        lr=hyper_params["LR_DQN"],
        eps=hyper_params["ADAM_EPS"],
        weight_decay=hyper_params["LAMBDA3"],
    )

    # make tuples to create an agent
    models = (dqn, dqn_target)

    # create an agent
    agent = Agent(env, args, hyper_params, models, dqn_optim)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
