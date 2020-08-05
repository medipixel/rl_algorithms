# -*- coding: utf-8 -*-
"""Train or test algorithms on PongNoFrameskip-v4.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import datetime

from rl_algorithms import build_agent
from rl_algorithms.common.env.atari_wrappers import atari_env_generator
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.utils import Config


def parse_args() -> argparse.Namespace:
    # configurations
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")
    parser.add_argument(
        "--seed", type=int, default=161, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="./configs/pong_no_frameskip_v4/dqn.py",
        help="config path",
    )
    parser.add_argument(
        "--test", dest="test", action="store_true", help="test mode (no training)"
    )
    parser.add_argument(
        "--grad-cam",
        dest="grad_cam",
        action="store_true",
        help="test mode with viewing Grad-CAM",
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
        "--off-worker-render",
        dest="worker_render",
        action="store_false",
        help="turn off worker rendering",
    )
    parser.add_argument(
        "--off-logger-render",
        dest="logger_render",
        action="store_false",
        help="turn off logger rendering",
    )
    parser.add_argument(
        "--render-after",
        type=int,
        default=0,
        help="start rendering after the input number of episode",
    )
    parser.add_argument(
        "--worker-verbose",
        dest="worker_verbose",
        action="store_true",
        help="turn on worker print statements",
    )
    parser.add_argument(
        "--log", dest="log", action="store_true", help="turn on logging"
    )
    parser.add_argument("--save-period", type=int, default=20, help="save model period")
    parser.add_argument(
        "--episode-num", type=int, default=500, help="total episode num"
    )
    parser.add_argument(
        "--max-update-step", type=int, default=100000, help="max update step"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=None, help="max episode step"
    )
    parser.add_argument(
        "--interim-test-num", type=int, default=5, help="interim test number"
    )
    parser.add_argument(
        "--integration-test",
        dest="integration_test",
        action="store_true",
        help="indicate integration test",
    )
    parser.add_argument(
        "--off-framestack",
        dest="framestack",
        action="store_false",
        help="turn off framestack",
    )
    parser.add_argument(
        "--save-pixel-gradient",
        dest="save_pixel_gradient",
        action="store_false",
        help="turn off framestack",
    )
    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()

    # env initialization
    env_name = "PongNoFrameskip-v4"
    env = atari_env_generator(
        env_name, args.max_episode_steps, frame_stack=args.framestack
    )

    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    NOWTIMES = datetime.datetime.now()
    curr_time = NOWTIMES.strftime("%y%m%d_%H%M%S")

    cfg = Config.fromfile(args.cfg_path)

    # If running integration test, simplify experiment
    if args.integration_test:
        cfg = common_utils.set_cfg_for_intergration_test(cfg)

    cfg.agent.env_info = dict(
        name="PongNoFrameskip-v4",
        observation_space=env.observation_space,
        action_space=env.action_space,
        is_discrete=True,
        is_atari=True,
    )
    cfg.agent.log_cfg = dict(agent=cfg.agent.type, curr_time=curr_time)
    build_args = dict(args=args, env=env)

    agent = build_agent(cfg.agent, build_args)

    import natsort
    import os
    import pickle

    # from tsne import bh_sne
    from MulticoreTSNE import MulticoreTSNE as TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    feature_list = []
    label_list = []

    save_dir = "rl_algorithms/pixel_grad/08_05_2020_11_27_10/state/"
    x_list = os.listdir(save_dir)
    x_list = natsort.natsorted(x_list)
    for _, x_dir in enumerate(x_list):
        with open(save_dir + x_dir, "rb") as f:
            state = pickle.load(f)
        state = np.array(state)
        feature = agent.learner.dqn.forward_feature(state).detach().cpu().numpy()
        q_value = np.max(
            agent.learner.dqn(agent._preprocess_state(state)).detach().cpu().numpy()
        )
        feature_list.append(np.squeeze(feature))
        label_list.append(q_value)

    feature_list = np.stack(feature_list, axis=0).astype("float64")
    # label_list = np.expand_dims(np.array(label_list), axis=-1)
    print("bh_sne processing..")
    output_2d = TSNE(n_jobs=8).fit_transform(feature_list)
    np.save("./rl_algorithms/tsne_result/output_2d.npy", output_2d, allow_pickle=False)

    plt.rcParams["figure.figsize"] = 20, 20
    # slice_len = len(x_list)
    # for i in zip(["red", "yellow", "blue", "purple", "green"], )
    plt.scatter(output_2d[:, 0], output_2d[:, 1], c=label_list, cmap="jet")
    plt.savefig("./rl_algorithms/tsne_result/output_2d.png", bbox_inches="tight")
    plt.show()
    print("test")

    # left data


if __name__ == "__main__":
    main()
