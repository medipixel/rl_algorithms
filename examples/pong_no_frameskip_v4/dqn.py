# -*- coding: utf-8 -*-
"""Run module for DQN on PongNoFrameskip-v4.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.env.atari_wrappers import atari_env_generator
from algorithms.common.env.multiprocessing_env import SubprocVecEnv
from algorithms.common.networks.cnn import CNN, CNNLayer
from algorithms.dqn.agent import Agent
from algorithms.dqn.networks import DuelingMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(1e4),  # openai baselines: int(1e4)
    "BATCH_SIZE": 32,  # 32 * n_workers, openai baselines: 32
    "LR_DQN": 6.25e-5,  # dueling: 6.25e-5
    "ADAM_EPS": 1.5e-4,  # rainbow: 1.5e-4
    "WEIGHT_DECAY": 0.0,  # this makes saturation in cnn weights
    "MAX_EPSILON": 1.0,
    "MIN_EPSILON": 0.02,
    "EPSILON_DECAY": 5e-6,
    "W_Q_REG": 1e-7,
    "PER_ALPHA": 0.6,  # openai baselines: 0.6
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
    "GRADIENT_CLIP": 10.0,  # dueling: 10.0
    "UPDATE_STARTS_FROM": int(1e4),  # openai baselines: int(1e4)
    "TRAIN_FREQ": 4,  # in openai baselines, train_freq = 4
    "MULTIPLE_LEARN": 1,
    "N_WORKERS": 1,
}


def run(env: gym.Env, env_name: str, args: argparse.Namespace):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # create multiple envs
    # configure environment so that it works for discrete actions
    env_single, n_envs = env, int(hyper_params["N_WORKERS"])
    env_list = [
        atari_env_generator(env_name, args.max_episode_steps) for _ in range(n_envs)
    ]
    env_multi = SubprocVecEnv(env_list)

    # create a model
    action_dim = env.action_space.n
    hidden_sizes = [512]

    def get_cnn_model():
        cnn_model = CNN(  # from rainbow
            cnn_layers=[
                CNNLayer(
                    input_size=4, output_size=32, kernel_size=8, stride=4, padding=1
                ),
                CNNLayer(input_size=32, output_size=64, kernel_size=4, stride=2),
                CNNLayer(input_size=64, output_size=64, kernel_size=3),
            ],
            fc_layers=DuelingMLP(
                input_size=3136, output_size=action_dim, hidden_sizes=hidden_sizes
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
        eps=hyper_params["ADAM_EPS"],
    )

    # make tuples to create an agent
    models = (dqn, dqn_target)

    # create an agent
    agent = Agent(env_single, env_multi, args, hyper_params, models, dqn_optim)
    agent.env_name = env_name

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
