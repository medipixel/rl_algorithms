from collections import deque
import random
from typing import Deque

import numpy as np

from rl_algorithms.common.helper_functions import get_n_step_info


def generate_dummy_buffer(maxlen: int, index: int) -> Deque:
    assert index <= maxlen
    n_step_buffer = deque(maxlen=maxlen)
    for i in range(maxlen):
        done = i == index
        transition = (np.array([i]), np.array([0]), i, np.array([i + 1]), done)
        n_step_buffer.append(transition)
    return n_step_buffer


def check_case1(maxlen: int):
    index = 0
    n_step_buffer = generate_dummy_buffer(maxlen, index)
    reward, next_state, _ = get_n_step_info(n_step_buffer, gamma=1)
    assert reward == index
    assert next_state == index + 1


def check_case2(maxlen: int):
    index = maxlen
    n_step_buffer = generate_dummy_buffer(maxlen, index)
    reward, next_state, _ = get_n_step_info(n_step_buffer, gamma=1)
    assert reward * 2 == maxlen * (maxlen - 1)
    assert next_state == maxlen


def check_case3(maxlen: int):
    index = random.randint(1, maxlen - 1)
    n_step_buffer = generate_dummy_buffer(maxlen, index)
    reward, next_state, _ = get_n_step_info(n_step_buffer, gamma=1)
    assert reward * 2 == index * (index + 1)
    assert next_state == index + 1


def test_get_n_step_info(maxlen=10):
    check_case1(maxlen)
    check_case2(maxlen)
    check_case3(maxlen)


#### n_step 정보가 잘 얻어지는 지 테스트
if __name__ == "__main__":
    test_get_n_step_info(maxlen=10)
