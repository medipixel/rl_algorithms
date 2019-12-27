import argparse
import datetime

import gym

from algorithms import DDPGAgent, build_agent
from algorithms.utils import Config


def test_config_registry():
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
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
    args = parser.parse_args()

    # set env
    env = gym.make("LunarLanderContinuous-v2")

    # check start time
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

    cfg = Config.fromfile(args.cfg_path)
    cfg.agent["log_cfg"] = dict(
        env="lunarlander_continuous_v2", agent=cfg.agent.type, curr_time=curr_time
    )
    default_args = dict(args=args, env=env)
    agent = build_agent(cfg.agent, default_args)
    assert isinstance(agent, DDPGAgent)


if __name__ == "__main__":
    test_config_registry()
