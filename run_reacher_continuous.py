# -*- coding: utf-8 -*-
"""Train or test baselines on Reacher-v2 of Mujoco.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

import argparse
import importlib

import gym

import algorithms.common.helper_functions as common_utils

# configurations
parser = argparse.ArgumentParser(description="Pytorch RL baselines")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility"
)
parser.add_argument("--algo", type=str, default="ddpg", help="choose an algorithm")
parser.add_argument(
    "--test", dest="test", action="store_true", help="test mode (no training)"
)
parser.add_argument(
    "--load-from", type=str, help="load the saved model and optimizer at the beginning"
)
parser.add_argument(
    "--off-render", dest="render", action="store_false", help="turn off rendering"
)
parser.add_argument(
    "--render-after",
    type=int,
    default=0,
    help="start rendering after the input number of episode",
)
parser.add_argument("--log", dest="log", action="store_true", help="turn on logging")
parser.add_argument("--save-period", type=int, default=200, help="save model period")
parser.add_argument("--episode-num", type=int, default=3000, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=-1, help="max episode step"
)
parser.add_argument(
    "--demo-path",
    type=str,
    default="data/lunarlander_continuous_demo.pkl",
    help="demonstration path",
)

parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=True)
parser.set_defaults(log=False)
args = parser.parse_args()


def main():
    """Main."""
    # env initialization
    env = gym.make("Reacher-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    module_path = "examples.Reacher-v2." + args.algo
    example = importlib.import_module(module_path)
    example.run(env, args, state_dim, action_dim)


if __name__ == "__main__":
    main()
