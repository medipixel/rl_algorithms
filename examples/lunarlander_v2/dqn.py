# -*- coding: utf-8 -*-
"""Run module for DQN on LunarLander-v2.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.dqn.agent import Agent
from algorithms.dqn.networks import IQNMLP, DuelingMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "N_STEP": 3,
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "W_N_STEP": 1.0,
    "W_Q_REG": 1e-7,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 32,
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
    "USE_DIST_Q": True,
    "N_TAU_SAMPLES": 64,
    "N_TAU_PRIME_SAMPLES": 64,
    "N_QUANTILE_SAMPLES": 32,
    "QUANTILE_EMBEDDING_DIM": 64,
    "KAPPA": 1.0,
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
        hidden_sizes = [512, 128]

        if hyper_params["USE_DIST_Q"]:
            model = IQNMLP(
                input_size=state_dim,
                output_size=action_dim,
                hidden_sizes=hidden_sizes,
                n_quantiles=hyper_params["N_QUANTILE_SAMPLES"],
                quantile_embedding_dim=hyper_params["QUANTILE_EMBEDDING_DIM"],
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
        weight_decay=hyper_params["WEIGHT_DECAY"],
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
