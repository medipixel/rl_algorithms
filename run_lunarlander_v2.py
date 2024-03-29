# -*- coding: utf-8 -*-
"""Train or test algorithms on LunarLander-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import datetime

import gym

from rl_algorithms import build_agent
import rl_algorithms.common.env.utils as env_utils
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.utils import YamlConfig


def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
    parser.add_argument(
        "--seed", type=int, default=777, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--integration-test",
        dest="integration_test",
        action="store_true",
        help="for integration test",
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./configs/lunarlander_v2/dqn.yaml",
        help="config path",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default=None,
        help="load the saved model and optimizer at the beginning",
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
    parser.add_argument(
        "--log", dest="log", action="store_true", help="turn on logging"
    )
    parser.add_argument(
        "--save-period", type=int, default=100, help="save model period"
    )
    parser.add_argument(
        "--episode-num", type=int, default=1500, help="total episode num"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=300, help="max episode step"
    )
    parser.add_argument(
        "--interim-test-num",
        type=int,
        default=10,
        help="number of test during training",
    )

    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()

    # env initialization
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    env, max_episode_steps = env_utils.set_env(env, args.max_episode_steps)

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

    cfg = YamlConfig(dict(agent=args.cfg_path)).get_config_dict()

    # If running integration test, simplify experiment
    if args.integration_test:
        cfg = common_utils.set_cfg_for_intergration_test(cfg)

    env_info = dict(
        name=env.spec.id,
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_atari=False,
    )
    log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time, cfg_path=args.cfg_path)
    build_args = dict(
        env=env,
        env_info=env_info,
        log_cfg=log_cfg,
        is_test=args.test,
        load_from=args.load_from,
        is_render=args.render,
        render_after=args.render_after,
        is_log=args.log,
        save_period=args.save_period,
        episode_num=args.episode_num,
        max_episode_steps=max_episode_steps,
        interim_test_num=args.interim_test_num,
    )
    agent = build_agent(cfg.agent, build_args)

    if not args.test:
        agent.train()
    else:
        agent.test()


if __name__ == "__main__":
    main()
