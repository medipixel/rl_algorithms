from typing import List, Tuple

import numpy as np
from scipy.stats import chisquare

from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer


def generate_transition(idx: int) -> Tuple[np.ndarray, ...]:
    """Make dummy transition for testing buffer."""
    obs = np.array([0])
    act = np.array([0])
    reward = idx
    next_obs = np.array([0])
    done = False
    return (obs, act, reward, next_obs, done)


def generate_sample_idx(buffer: ReplayBuffer) -> int:
    """Generate indices to test whether sampled uniformly or not."""
    for i in range(buffer.max_len):
        buffer.add(generate_transition(i))
    _, _, idx, _, _ = buffer.sample()
    return idx


def check_uniform(lst: List) -> bool:
    """Check the distribution is Uniform Distribution."""
    res = chisquare(lst)
    return res[1] >= 0.05


def test_uniform_sample(buffer_length=32, batch_size=8):
    """Test whether transitions are uniformly sampled from replay buffer."""

    n_repeat = 10000  # number of repetition of sample experiments

    buffer = ReplayBuffer(max_len=buffer_length, batch_size=batch_size)

    sampled_lst = [0] * buffer.max_len  # make list to count sampled index
    for _ in range(n_repeat):  # sampling index for the n_repeat times
        indices = generate_sample_idx(buffer)  # sample index from buffer
        for idx in indices:
            sampled_lst[int(idx)] += 1 / n_repeat  # make frequence to proportion

    assert check_uniform(sampled_lst), "This distribution is not uniform."


if __name__ == "__main__":
    test_uniform_sample()
