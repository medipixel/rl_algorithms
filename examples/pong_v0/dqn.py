# -*- coding: utf-8 -*-
"""Run module for DQN on Pong-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import multiprocessing

import gym
import torch
import torch.optim as optim

import algorithms.common.env.utils as env_utils
from algorithms.common.env.utils import env_generator, make_envs
from algorithms.common.helper_functions import identity
from algorithms.common.networks.cnn import CNN, CNNLayer
from algorithms.dqn.agent import Agent
from examples.pong_v0.wrappers import WRAPPERS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_cpu = multiprocessing.cpu_count()

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "W_Q_REG": 1e-7,
    "BUFFER_SIZE": int(2e5),
    "BATCH_SIZE": 64,
    "LR_DQN": 1e-4,
    "WEIGHT_DECAY": 1e-6,
    "MAX_EPSILON": 1.0,
    "MIN_EPSILON": 0.02,
    "EPSILON_DECAY": 3e-6,
    "PER_ALPHA": 0.5,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "UPDATE_STARTS_FROM": int(2e4),
    "MULTIPLE_LEARN": n_cpu // 2 if n_cpu >= 2 else 1,
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

    def get_cnn_model():
        # it consists of only cnn layers
        cnn_model = CNN(
            cnn_layers=[
                CNNLayer(input_size=4, output_size=32, kernel_size=8, stride=4),
                CNNLayer(input_size=32, output_size=64, kernel_size=4, stride=2),
                CNNLayer(input_size=64, output_size=64, kernel_size=3, stride=1),
                CNNLayer(input_size=64, output_size=512, kernel_size=7, stride=4),
                CNNLayer(
                    input_size=512,
                    output_size=action_dim,
                    kernel_size=1,
                    stride=1,
                    activation_fn=identity,
                ),
            ],
            fc_layers=identity,
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
