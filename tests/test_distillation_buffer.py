"""Test only one step of run file for training."""

import os
import pickle
import re
import shutil
import subprocess


def check_train_phase_data_generating(config: str, run_file: str):
    cmd = (
        f"python {run_file} --cfg-path {config} --integration-test "
        + f"--episode-num 1 --interim-test 1 --off-render"
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
    data_pattern = r"./data/.+/"
    checkpoint_path = re.findall(pattern, str(output))[0]

    # check if data saved properly
    num_episode_step = re.findall(r"episode step: \d+", str(output))[0]
    num_episode_step = int(re.findall(r"\d+", num_episode_step)[0])

    # check if the number of data is same with iterated episode step
    data_path = re.findall(data_pattern, str(output))[0]
    saved_data_list = os.listdir(data_path)
    assert (
        len(saved_data_list) == num_episode_step
    ), "The number of data does not match the number of iterated episode steps."

    with open(data_path + saved_data_list[0], "rb") as f:
        datum = pickle.load(f)

    assert (
        len(datum) == 1
    ), "The length of the data is not appropriate(length must be 1, state only)."

    return checkpoint_path, data_path


def check_distillation_buffer(config: str, run_file: str):
    """Test that 1 episode of run file works well."""

    checkpoint_path, data_path = check_train_phase_data_generating(config, run_file)
    check_save_path(checkpoint_path)
    check_save_path(data_path)


def check_save_path(save_path: str):
    """Check checkpoint that tested run file makes and remove the checkpoint."""
    assert os.path.exists(save_path)

    # Remove checkpoint dir
    shutil.rmtree(save_path)


def test_distillation():
    """Test distillation buffer"""

    check_distillation_buffer(
        "configs/pong_no_frameskip_v4/distillation_dqn.py",
        "run_pong_no_frameskip_v4.py",
    )
    check_distillation_buffer(
        "configs/lunarlander_v2/distillation_dqn.py", "run_lunarlander_v2.py"
    )


if __name__ == "__main__":
    test_distillation()
