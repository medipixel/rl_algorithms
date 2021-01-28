import argparse
import datetime

import gym

from rl_algorithms import build_agent
from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.utils import Config


def parse_args(args: list):
    parser = argparse.ArgumentParser(description="Pytorch RL rl_algorithms")
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./configs/lunarlander_continuous_v2/ddpg.py",
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
    return parser.parse_args(args)


def test_config_registry():
    # configurations
    args = parse_args(["--test"])

    # set env
    env = gym.make("LunarLanderContinuous-v2")

    # check start time
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

    cfg = Config.fromfile(args.cfg_path)
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
        max_episode_steps=args.max_episode_steps,
        interim_test_num=args.interim_test_num,
    )
    agent = build_agent(cfg.agent, build_args)
    assert isinstance(agent, Agent)


if __name__ == "__main__":
    test_config_registry()
