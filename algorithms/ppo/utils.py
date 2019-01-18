# -*- coding: utf-8 -*-
"""Utility functions for PPO.

This module has PPO util functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

import numpy as np
import torch


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def decompose_memory(memory):
    """Decompose states, actions, rewards, dones from the memory."""
    memory = np.array(memory)
    states = torch.from_numpy(np.vstack(memory[:, 0])).float().to(device)
    log_probs = torch.from_numpy(np.vstack(memory[:, 1])).float().to(device)
    actions = torch.from_numpy(np.vstack(memory[:, 2])).float().to(device)
    rewards = torch.from_numpy(np.vstack(memory[:, 3])).float().to(device)
    dones = torch.from_numpy(
                np.vstack(memory[:, 4]).astype(np.uint8)).float().to(device)

    return states, log_probs, actions, rewards, dones


def ppo_iter(epoch, mini_batch_size, states,
             log_probs, actions, returns, advantages):
    """Yield mini-batches."""
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], log_probs[rand_ids, :], \
                actions[rand_ids, :], returns[rand_ids, :], \
                advantages[rand_ids, :]


# taken from https://github.com/ikostrikov/pytorch-trpo
def get_gae(rewards, values, dones, gamma, lambd):
    """Calculate returns and GAEs."""
    masks = 1 - dones
    returns = torch.zeros_like(rewards)
    deltas = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * lambd * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    # normalize advantages
    advantages = (advantages - advantages.mean()) /\
                 (advantages.std() + 1e-7)

    return returns, advantages
