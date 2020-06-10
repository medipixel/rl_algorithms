import argparse
import datetime

import gym

from rl_algorithms import build_agent
from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.utils import Config


def parse_args(args: list):
    parser = argparse.ArgumentParser(description="Pytorch RL rl_algorithms")
    parser.add_argument(
        "--load-from",
        default=None,
        type=str,
        help="load the saved model and optimizer at the beginning",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./configs/lunarlander_continuous_v2/ddpg.py",
        help="config path",
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
    cfg.agent.env_info = dict(
        env_name="LunarLanderContinuous-v2",
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_discrete=False,
    )
    cfg.agent.log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time)
    default_args = dict(args=args, env=env)
    agent = build_agent(cfg.agent, default_args)
    assert isinstance(agent, Agent)


if __name__ == "__main__":
    test_config_registry()
