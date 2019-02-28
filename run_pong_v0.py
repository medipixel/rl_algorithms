# -*- coding: utf-8 -*-
"""Train or test algorithms on Pong-v0.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import importlib

import gym

import algorithms.common.helper_functions as common_utils

# configurations
parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility"
)
parser.add_argument("--algo", type=str, default="dqn", help="choose an algorithm")
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
parser.add_argument("--save-period", type=int, default=25, help="save model period")
parser.add_argument("--episode-num", type=int, default=500, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=-1, help="max episode step"
)

parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=True)
parser.set_defaults(log=False)
args = parser.parse_args()


def main():
    """Main."""
    # env initialization
    env = gym.make("Pong-v0")

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    module_path = "examples.pong_v0." + args.algo
    example = importlib.import_module(module_path)
    example.run(env, args)


if __name__ == "__main__":
    main()
