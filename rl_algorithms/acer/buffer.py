# TODO : Move to common buffer

from collections import deque
import random

import torch


class ReplayMemory:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size, on_policy=False):
        if on_policy:
            mini_batch = [self.memory[-1]]
        else:
            mini_batch = random.sample(self.memory, batch_size)

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                state, action, reward, prob, done_mask = transition

                s_lst.append(state)
                a_lst.append([action])
                r_lst.append(reward)
                prob_lst.append(prob)
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False

        state, action, reward, prob, done_mask, is_first = (
            torch.FloatTensor(s_lst),
            torch.LongTensor(a_lst),
            r_lst,
            torch.FloatTensor(prob_lst),
            done_lst,
            is_first_lst,
        )
        return state, action, reward, prob, done_mask, is_first

    def __len__(self):
        return len(self.memory)
