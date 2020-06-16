import os


def check_run(config_root: str, run_file: str):
    """"""
    # loop of configs
    configs = os.listdir(config_root)
    for cfg in configs:
        # except such as __init__, __pycache__
        if "__" in cfg:
            continue
        print(cfg)

        res = os.system(
            f"python {run_file} "
            + f"--cfg-path {config_root}{cfg} "
            + f"--off-render --episode-num 1 --max-episode-step 1 --seed 12345"
        )

        assert res == 0


def test_run_lunarlander_continuous():
    """Test all agents that train LunarLanderContinuous-v2 env."""
    check_run("configs/lunarlander_continuous_v2/", "run_lunarlander_continuous_v2.py")


def test_run_lunarlander():
    """Test all agents that train LunarLander-v2 env."""
    check_run("configs/lunarlander_v2/", "run_lunarlander_v2.py")


def test_run_pong_no_frame_skip():
    """Test all agents that train PongNoFrameskip-v4 env."""
    check_run("configs/pong_no_frameskip_v4/", "run_pong_no_frameskip_v4.py")


if __name__ == "__main__":
    test_run_lunarlander_continuous()
    test_run_lunarlander()
    test_run_pong_no_frame_skip()
