# -*- coding: utf-8 -*-
"""Train or test algorithms on PongNoFrameskip-v4.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import importlib

from algorithms.common.env.atari_wrappers import atari_env_generator
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
parser.add_argument("--save-period", type=int, default=20, help="save model period")
parser.add_argument("--episode-num", type=int, default=1500, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=None, help="max episode step"
)
parser.add_argument(
    "--interim-test-num", type=int, default=10, help="interim test number"
)
parser.add_argument("--wandb-project", type=str, default="", help="wandb project name")

parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=True)
parser.set_defaults(log=False)
args = parser.parse_args()


def main():
    """Main."""
    # env initialization
    env_name = "PongNoFrameskip-v4"
    env = atari_env_generator(env_name, args.max_episode_steps)

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    module_path = "examples.pong_no_frameskip_v4." + args.algo
    example = importlib.import_module(module_path)
    example.run(env, env_name, args)


if __name__ == "__main__":
    main()
