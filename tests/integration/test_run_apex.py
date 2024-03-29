"""Test only one step of run file for training."""

import os
import os.path as osp
import re
import shutil
import subprocess


def check_run_apex(config_root: str, run_file: str):
    """Test that 1 episode of run file works well."""
    test_dir = osp.dirname(osp.abspath(__file__))
    pkg_root_dir = osp.dirname(osp.dirname(test_dir))
    os.chdir(pkg_root_dir)

    # loop of configs
    configs = os.listdir(config_root)
    for cfg in configs:
        # except such as __init__, __pycache__
        if "__" in cfg or "apex" not in cfg:
            continue

        cmd = (
            f"python {run_file} --cfg-path {config_root}{cfg} --integration-test "
            + f"--off-render --seed 12345 --interim-test-num 1"
        )

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True,
        )
        output, _ = p.communicate()
        print(str(output))
        assert p.returncode == 0

        # Find saved checkpoint path
        pattern = r"./checkpoint/.+/"
        save_path = re.findall(pattern, str(output))[0]
        print(save_path)

        check_save_path(save_path)


def check_save_path(save_path: str):
    """Check checkpoint that tested run file makes and remove the checkpoint."""
    assert os.path.exists(save_path)

    # Remove checkpoint dir
    shutil.rmtree(save_path)


def test_run_pong_no_frame_skip():
    """Test all agents that train PongNoFrameskip-v4 env."""
    check_run_apex("configs/pong_no_frameskip_v4/", "run_pong_no_frameskip_v4.py")


if __name__ == "__main__":
    test_run_pong_no_frame_skip()
