import os
import pickle
import random
import shutil

import numpy as np
import pytest

from rl_algorithms.common.buffer.distillation_buffer import DistillationBuffer

FOLDER_PATH_LIST = [
    "data/distillation_buffer/test/expert_data/",
    "data/distillation_buffer/test/expert_data2/",
    "data/distillation_buffer/test/trainphase_data/",
]


def gen_test_data(num_files: int):
    """Generate dummy data."""
    for _dir in FOLDER_PATH_LIST:
        os.makedirs(_dir)

    for i, _dir in enumerate(FOLDER_PATH_LIST):
        for j in range(num_files):
            state = np.random.randint(0, 255, size=(3, 3, 2), dtype=np.uint8)
            action = np.zeros(3)
            action[random.randint(0, len(action) - 1)] = 1
            action = action.astype(np.int)
            if "trainphase" in _dir:
                with open(f"{FOLDER_PATH_LIST[i]}{j:07}.pkl", "wb") as f:
                    pickle.dump([state], f)
            else:
                with open(f"{FOLDER_PATH_LIST[i]}{j:07}.pkl", "wb") as f:
                    pickle.dump([state, action], f)


def check_multiple_data_load(num_files: int):
    """Check if DistillationBuffer can load data from multiple path."""
    batch_size = num_files * len(FOLDER_PATH_LIST[:-1])
    memory = DistillationBuffer(batch_size, FOLDER_PATH_LIST[:-1], 20202020,)
    memory.reset_dataloader()
    state, _ = memory.sample_for_diltillation()
    assert state.shape[0] == batch_size


def check_mixture_data_assert(num_files: int):
    """Check if DistillationBuffer can check whether trainphase & expert data is mixed."""
    memory = DistillationBuffer(num_files, FOLDER_PATH_LIST, 20202020,)
    with pytest.raises(AssertionError, match=r"mixture"):
        memory.reset_dataloader()


def delete_path(path: str):
    """Delete directory."""
    shutil.rmtree(path)


def test_distillation_buffer():
    """Test DistillationBuffer."""
    try:
        num_file = 7
        gen_test_data(num_file)
        check_multiple_data_load(num_file)
        check_mixture_data_assert(num_file)

    except Exception as e:
        raise e

    finally:
        delete_path("data/distillation_buffer/test")


if __name__ == "__main__":
    test_distillation_buffer()
