# -*- coding: utf-8 -*-
"""Run module for DQN on CarRacing-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import multiprocessing

import gym
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.common.env.utils import env_generator, make_envs
from algorithms.common.networks.cnn import CNN, CNNLayer
from algorithms.common.networks.mlp import MLP
from algorithms.dqn.agent import Agent
from examples.car_racing_v0.wrappers import Continuous2Discrete, PreprocessedObservation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_cpu = multiprocessing.cpu_count()

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "W_Q_REG": 1e-7,
    "BUFFER_SIZE": int(4e5),
    "BATCH_SIZE": 128,
    "LR_DQN": 1e-3,
    "WEIGHT_DECAY": 1e-6,
    "MAX_EPSILON": 1.0,
    "MIN_EPSILON": 0.01,
    "EPSILON_DECAY": 1e-5,
    "PER_ALPHA": 0.5,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "UPDATE_STARTS_FROM": int(1e4),
    "MULTIPLE_LEARN": n_cpu // 2 if n_cpu >= 2 else 1,
    "N_WORKERS": n_cpu,
}


def run(env: gym.Env, args: argparse.Namespace, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # create multiple envs
    # configure environment so that it works for discrete actions
    env_single = Continuous2Discrete(PreprocessedObservation(env))
    env_wrappers = [PreprocessedObservation, Continuous2Discrete]
    env_gen = env_generator("CarRacing-v0", args, env_wrappers)
    env_multi = make_envs(env_gen, n_envs=hyper_params["N_WORKERS"])

    action_dim = env_single.action_dim  # get discrete action dimension

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
    agent = Agent(env_single, env_multi, args, hyper_params, models, dqn_optim)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
