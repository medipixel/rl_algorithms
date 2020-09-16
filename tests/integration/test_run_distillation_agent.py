"""Test only one step of distillation file for training."""

import os
import pickle
import re
import shutil
import subprocess


def check_distillation_agent(config: str, run_file: str):
    """Test that 1 episode of run file works well."""
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

    # Find saved checkpoint path and data path.
    pattern = r"./checkpoint/.+/"
    data_pattern = r"./data/.+/"
    checkpoint_path = re.findall(pattern, str(output))[0]
    full_data_path, n_frame_from_last_path = re.findall(data_pattern, str(output))

    try:
        num_episode_step = re.findall(r"episode step: \d+", str(output))[0]
        num_episode_step = int(re.findall(r"\d+", num_episode_step)[0])

        # Check if the number of data is same with iterated episode step.
        saved_data_list = os.listdir(full_data_path)
        assert (
            len(saved_data_list) == num_episode_step
        ), "The number of data does not match the number of iterated episode steps."

        # Check if n_frame_from_last works well.
        n_frame_from_last_data_list = os.listdir(n_frame_from_last_path)
        assert 3 == len(
            n_frame_from_last_data_list
        ), f"n_frame_from_last doesn't work properly(expected num of data: 3, num of data: {len(n_frame_from_last_data_list)})."

        # Check if train-phase data only contaions state, not state & q value.
        with open(full_data_path + saved_data_list[0], "rb") as f:
            datum = pickle.load(f)
        assert (
            len(datum) == 1
        ), "The length of the data is not appropriate(length must be 1, state only)."

    except Exception as e:
        raise e

    finally:
        """Delete generated directories."""
        delete_path(checkpoint_path)
        delete_path(full_data_path)
        delete_path(n_frame_from_last_path)


def delete_path(path: str):
    """Delete directory."""
    shutil.rmtree(path)


# TODO: Add student training test code.
def test_distillation():
    """Test distillation agent."""
    check_distillation_agent(
        "configs/pong_no_frameskip_v4/distillation_dqn.py",
        "run_pong_no_frameskip_v4.py",
    )
    check_distillation_agent(
        "configs/lunarlander_v2/distillation_dqn.py", "run_lunarlander_v2.py"
    )


if __name__ == "__main__":
    test_distillation()
