# -*- coding: utf-8 -*-
"""Run module for DQN on Pong-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch
import torch.nn as nn
import torch.optim as optim

import algorithms.common.env.utils as env_utils
from algorithms.common.env.utils import env_generator, make_envs
from algorithms.common.networks.cnn import CNNLayer
from algorithms.dqn.agent import Agent
from algorithms.dqn.networks import DuelingCNN, DuelingMLP
from examples.pong_v0.wrappers import WRAPPERS

# import multiprocessing


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_cpu = 4  # multiprocessing.cpu_count()

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 64,
    "LR_DQN": 1e-4,
    "WEIGHT_DECAY": 1e-6,
    "MAX_EPSILON": 1.0,
    "MIN_EPSILON": 0.02,
    "EPSILON_DECAY": 1e-5,
    "W_Q_REG": 1e-7,
    "PER_ALPHA": 0.6,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "GRADIENT_CLIP": 0.5,
    "UPDATE_STARTS_FROM": int(2e4),
    "MULTIPLE_LEARN": n_cpu,
    "N_WORKERS": n_cpu,
}


def run(env: gym.Env, args: argparse.Namespace):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # create multiple envs
    # configure environment so that it works for discrete actions
    env_single = env_utils.set_env(env, args, WRAPPERS)
    env_gen = env_generator("Pong-v0", args, WRAPPERS)
    env_multi = make_envs(env_gen, n_envs=hyper_params["N_WORKERS"])

    # create a model
    action_dim = env.action_space.n
    hidden_sizes = [256, 256]

    def get_cnn_model():
        cnn_model = DuelingCNN(
            cnn_layers=[
                CNNLayer(
                    input_size=4,
                    output_size=32,
                    kernel_size=5,
                    post_activation_fn=nn.MaxPool2d(3),
                ),
                CNNLayer(
                    input_size=32,
                    output_size=32,
                    kernel_size=3,
                    post_activation_fn=nn.MaxPool2d(3),
                ),
                CNNLayer(
                    input_size=32,
                    output_size=64,
                    kernel_size=2,
                    post_activation_fn=nn.MaxPool2d(3),
                ),
            ],
            fc_layers=DuelingMLP(
                input_size=256, output_size=action_dim, hidden_sizes=hidden_sizes
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
    agent.env_name = "Pong-v0"

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
