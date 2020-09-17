import numpy as np
from scipy.stats import chisquare

from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer


def generate_transition(idx):
    "Make toy transition for testing buffer."
    obs = np.array([0])
    act = np.array([0])
    reward = idx
    next_obs = np.array([0])
    done = False
    return (obs, act, reward, next_obs, done)


def generate_sample_idx(buffer_length, batch_size):
    "Generate indices to test whether sampled uniformly or not."
    toy_buffer = ReplayBuffer(max_len=buffer_length, batch_size=batch_size)
    for i in range(buffer_length):
        toy_buffer.add(generate_transition(i))
    _, _, idx, _, _ = toy_buffer.sample()
    return idx


def cal_sample_portion(buffer_length, batch_size, times):
    sampled_freq_lst = [0] * buffer_length
    for _ in range(times):
        indices = generate_sample_idx(buffer_length, batch_size)
        for idx in indices:
            sampled_freq_lst[int(idx)] += 1 / times
    return sampled_freq_lst


def check_uniform(lst):
    "Check the distribution is Uniform Distribution."
    res = chisquare(lst)
    return res[1] >= 0.975


def test_uniform(buffer_length=32, batch_size=8):
    "Test whether transitions are uniformly sampled from replay buffer."
    n_repeat = 10000  # number of repetition of sample experiments
    portion_lst = cal_sample_portion(
        buffer_length, batch_size, n_repeat
    )  # make portion list of sampling

    assert check_uniform(portion_lst), "This distribution is not uniform."


if __name__ == "__main__":
    test_uniform(buffer_length=32, batch_size=8)
