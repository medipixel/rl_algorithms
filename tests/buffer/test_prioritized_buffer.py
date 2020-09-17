from typing import List, Tuple

import numpy as np
from scipy.stats import ks_2samp

from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper


def generate_prioritized_buffer(
    buffer_length: int, batch_size: int, idx_lst=None, prior_lst=None
) -> Tuple[PrioritizedBufferWrapper, List]:
    """Generate Prioritized Replay Buffer with random Prior."""
    buffer = ReplayBuffer(max_len=buffer_length, batch_size=batch_size)
    prioritized_buffer = PrioritizedBufferWrapper(buffer)
    priority = np.random.randint(
        10, size=buffer_length
    )  # generate base-priority randomly
    for i, j in enumerate(priority):
        prioritized_buffer.sum_tree[i] = j
    if idx_lst:  # if idx_lst and prior_lst exist, then update priorty
        for i, j in list(zip(idx_lst, prior_lst)):
            priority[i] = j
            prioritized_buffer.sum_tree[i] = j

    prop_lst = [i / sum(priority) for i in priority]

    return prioritized_buffer, prop_lst


def sample_dummy(prioritized_buffer: PrioritizedBufferWrapper, times: int) -> List:
    """Sample from prioritized buffer and Return indices."""
    assert isinstance(prioritized_buffer, PrioritizedBufferWrapper)

    sampled_lst = [0] * prioritized_buffer.buffer.max_len
    for _ in range(times):
        indices = prioritized_buffer._sample_proportional(
            prioritized_buffer.buffer.batch_size
        )
        for idx in indices:
            sampled_lst[idx] += 1 / (times * prioritized_buffer.buffer.batch_size)
    return sampled_lst


def check_prioritized(prop_lst: List, sampled_lst: List) -> bool:
    """Check two input lists have same distribution by kstest.

        Reference:
        https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """
    res = ks_2samp(prop_lst, sampled_lst)
    return res[1] >= 0.05


def test_prioritized(buffer_length=32, batch_size=4):
    """Test whether transitions are prioritized sampled from replay buffer."""

    n_repeat = 1000
    idx_lst = [0, 1, 2, 3]  # index that we want to manipulate
    prior_lst = [100, 10, 1, 1]  # prior that we want to manipulate

    # generate prioitized buffer, return buffer and its proportion
    buffer, prop = generate_prioritized_buffer(
        buffer_length, batch_size, idx_lst, prior_lst
    )
    assert isinstance(buffer, PrioritizedBufferWrapper)
    sampled_lst = [0] * buffer.buffer.max_len  # make list to count sampled index
    for _ in range(n_repeat):  # sampling index for the n_repeat times
        # sample index from buffer
        indices = buffer._sample_proportional(buffer.buffer.batch_size)
        for idx in indices:
            sampled_lst[idx] += 1 / (
                n_repeat * buffer.buffer.batch_size
            )  # make frequence to proportion

    assert check_prioritized(prop, sampled_lst), "Two distributions are different."


if __name__ == "__main__":
    test_prioritized()
