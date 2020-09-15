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


def gen_test_data():
    """Generate dummy data."""
    for _dir in FOLDER_PATH_LIST:
        os.makedirs(_dir)
    num_file = 21
    for i, _dir in enumerate(FOLDER_PATH_LIST):
        for j in range(num_file):
            state = np.random.randint(0, 255, size=(3, 3, 2), dtype=np.uint8)
            action = np.zeros(3)
            action[random.randint(0, len(action) - 1)] = 1
            action = action.astype(np.int)
            if "trainphase" in _dir:
                with open(FOLDER_PATH_LIST[i] + f"{j:07}.pkl", "wb") as f:
                    pickle.dump([state], f)
            else:
                with open(FOLDER_PATH_LIST[i] + f"{j:07}.pkl", "wb") as f:
                    pickle.dump([state, action], f)


def check_multiple_data_load():
    """Check if DistillationBuffer can load data from multiple path."""
    memory = DistillationBuffer(42, FOLDER_PATH_LIST[:-1], 20202020,)
    memory.reset_dataloader()
    state, _ = memory.sample_for_diltillation()
    assert state.shape[0] == 42


def check_mixture_data_assert():
    """Check if DistillationBuffer can check whether trainphase & expert data is mixed."""
    memory = DistillationBuffer(21, FOLDER_PATH_LIST, 20202020,)
    with pytest.raises(AssertionError, match=r"mixture"):
        memory.reset_dataloader()


def test_distillation_buffer():
    """Test DistillationBuffer."""
    try:
        gen_test_data()
        check_multiple_data_load()
        check_mixture_data_assert()

    except Exception as e:
        raise e

    finally:
        shutil.rmtree("data/distillation_buffer/test")


if __name__ == "__main__":
    test_distillation_buffer()
