import numpy as np
from scipy.stats import ks_2samp

from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper


def generate_randomly_prioritized_buffer(buffer_length, batch_size):
    "Generate Prioritized Replay Buffer with random Prior."
    toy_buffer = ReplayBuffer(max_len=buffer_length, batch_size=batch_size)
    toy_prioritized_buffer = PrioritizedBufferWrapper(toy_buffer)
    priority = np.random.randint(1000, size=buffer_length)
    prop_lst = [i / sum(priority) for i in priority]
    for i, j in enumerate(priority):
        toy_prioritized_buffer.sum_tree[i] = j
    return toy_prioritized_buffer, prop_lst


def sample_toy(prioritized_buffer, times):
    "Sample from prioritized buffer and Return indices."
    assert isinstance(prioritized_buffer, PrioritizedBufferWrapper)

    sampled_lst = [0] * prioritized_buffer.buffer.max_len
    for _ in range(times):
        indices = prioritized_buffer._sample_proportional(
            prioritized_buffer.buffer.batch_size
        )
        for idx in indices:
            sampled_lst[idx] += 1 / (times * prioritized_buffer.buffer.batch_size)
    return sampled_lst


def check_prioritized(prop_lst, sampled_lst):
    """Check two input lists have same distribution by kstest.

        Reference:
        https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    """
    res = ks_2samp(prop_lst, sampled_lst)
    return res[1] >= 0.975


def test_prioritized(buffer_length=32, batch_size=16, times=1000):
    buffer, prop = generate_randomly_prioritized_buffer(buffer_length, batch_size)
    sampled_lst = sample_toy(buffer, times)
    assert check_prioritized(prop, sampled_lst), "Two distributions are different."


if __name__ == "__main__":
    test_prioritized(buffer_length=32, batch_size=16, times=1000)
