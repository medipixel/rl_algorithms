# -*- coding: utf-8 -*-
"""Run module for DQN on PongNoFrameskip-v4.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.networks.cnn import CNN, CNNLayer
from algorithms.dqn.agent import Agent
from algorithms.dqn.networks import CategoricalCNN, CategoricalDuelingMLP, DuelingMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "N_STEP": 3,
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(1e4),  # openai baselines: int(1e4)
    "BATCH_SIZE": 128,  # openai baselines: 32
    "LR_DQN": 1e-4,  # dueling: 6.25e-5, openai baselines: 1e-4
    "ADAM_EPS": 1e-8,  # rainbow: 1.5e-4, openai baselines: 1e-8
    "WEIGHT_DECAY": 0.0,  # this makes saturation in cnn weights
    "MAX_EPSILON": 1.0,
    "MIN_EPSILON": 0.01,  # openai baselines: 0.01
    "EPSILON_DECAY": 1e-6,  # openai baselines: 1e-7 / 1e-1
    "W_N_STEP": 1.0,
    "W_Q_REG": 0.0,
    "PER_ALPHA": 0.6,  # openai baselines: 0.6
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "GRADIENT_CLIP": 10.0,  # dueling: 10.0
    "UPDATE_STARTS_FROM": int(1e4),  # openai baselines: int(1e4)
    "TRAIN_FREQ": 4,  # in openai baselines, train_freq = 4
    "MULTIPLE_LEARN": 1,
    # C51
    "USE_C51": True,
    "V_MIN": -10,
    "V_MAX": 10,
    "ATOMS": 51,
}


def run(env: gym.Env, env_name: str, args: argparse.Namespace):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        env_name (str): Environment name of openAI Gym
        args (argparse.Namespace): arguments including training settings

    """

    def get_cnn_model():
        Model = CategoricalCNN if hyper_params["USE_C51"] else CNN
        fc_input_size = 3136
        hidden_sizes = [512]
        action_dim = env.action_space.n

        if hyper_params["USE_C51"]:
            Model = CategoricalCNN
            fc_model = CategoricalDuelingMLP(
                input_size=fc_input_size,
                action_size=action_dim,
                hidden_sizes=hidden_sizes,
                v_min=hyper_params["V_MIN"],
                v_max=hyper_params["V_MAX"],
                atom_size=hyper_params["ATOMS"],
            ).to(device)

        else:
            Model = CNN
            fc_model = DuelingMLP(
                input_size=fc_input_size,
                output_size=action_dim,
                hidden_sizes=hidden_sizes,
            ).to(device)

        # create a model
        cnn_model = Model(  # from rainbow
            cnn_layers=[
                CNNLayer(
                    input_size=4, output_size=32, kernel_size=8, stride=4, padding=1
                ),
                CNNLayer(input_size=32, output_size=64, kernel_size=4, stride=2),
                CNNLayer(input_size=64, output_size=64, kernel_size=3),
            ],
            fc_layers=fc_model,
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
        eps=hyper_params["ADAM_EPS"],
    )

    # make tuples to create an agent
    models = (dqn, dqn_target)

    # create an agent
    agent = Agent(env, args, hyper_params, models, dqn_optim)
    agent.env_name = env_name

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
