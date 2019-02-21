# -*- coding: utf-8 -*-
"""Train or test algorithms on LunarLander-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import importlib

import gym

import algorithms.common.env.utils as env_utils
import algorithms.common.helper_functions as common_utils

# configurations
parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility"
)
parser.add_argument("--algo", type=str, default="ppo", help="choose an algorithm")
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
parser.add_argument("--save-period", type=int, default=100, help="save model period")
parser.add_argument("--episode-num", type=int, default=1500, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=300, help="max episode step"
)

parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=True)
parser.set_defaults(log=False)
args = parser.parse_args()


def main():
    """Main."""
    # env initialization
    env = gym.make("LunarLander-v2")
    env_utils.set_env(env, args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # 0: none 1: left 2: engine 3: right

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    module_path = "examples.lunarlander_v2." + args.algo
    example = importlib.import_module(module_path)
    example.run(env, args, state_dim, action_dim)


if __name__ == "__main__":
    main()
