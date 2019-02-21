# -*- coding: utf-8 -*-
"""Run module for DQN on CarRacing-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common.networks.cnn import CNN, CNNLayer
from algorithms.common.networks.mlp import MLP
from algorithms.dqn.agent import Agent
from examples.car_racing_v0.wrappers import Continuous2Discrete, PreprocessedObservation

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
    "UPDATE_STARTS_FROM": 5000,
}


def run(env: gym.Env, args: argparse.Namespace, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # configure environment so that it works for discrete actions
    env = Continuous2Discrete(PreprocessedObservation(env))
    action_dim = env.action_dim  # get discrete action dimension

    # create a model
    def get_cnn_model():
        fc_hidden_sizes = [256, 256]

        cnn_model = CNN(
            cnn_layers=[
                CNNLayer(
                    input_size=1,
                    output_size=8,
                    kernel_size=7,
                    stride=3,
                    pulling_fn=nn.MaxPool2d(2, 2),
                ),
                CNNLayer(
                    input_size=8,
                    output_size=16,
                    kernel_size=3,
                    pulling_fn=nn.MaxPool2d(2, 2),
                ),
            ],
            fc_layers=MLP(
                input_size=576, output_size=action_dim, hidden_sizes=fc_hidden_sizes
            ),
        ).to(device)
        return cnn_model

    dqn = get_cnn_model()
    dqn_target = get_cnn_model()
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
