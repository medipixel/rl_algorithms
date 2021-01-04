# TODO : Move to common buffer

from collections import deque
import random

import torch

from rl_algorithms.common.abstract.buffer import BaseBuffer


class ReplayMemory(BaseBuffer):
    """Replay Memory for ACER
    (https://github.com/seungeunrho/minimalRL/blob/master/acer.py)
    """

    def __init__(self, buffer_size: int):
        self.memory = deque(maxlen=buffer_size)

    def add(self, transition: tuple):
        self.memory.append(transition)

    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.memory[-1]]
        else:
            mini_batch = random.sample(self.memory, 1)

        s_lst, a_lst, r_lst, prob_lst, done_lst = [], [], [], [], []
        for seq in mini_batch:
            for transition in seq:
                state, action, reward, prob, done_mask = transition

                s_lst.append(state)
                a_lst.append([action])
                r_lst.append(reward)
                prob_lst.append(prob)
                done_lst.append(done_mask)

        state, action, reward, prob, done_mask = (
            torch.FloatTensor(s_lst),
            torch.LongTensor(a_lst),
            r_lst,
            torch.FloatTensor(prob_lst),
            done_lst,
        )
        return state, action, reward, prob, done_mask

    def __len__(self):
        return len(self.memory)
