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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def decompose_memory(memory: list):
    """Decompose states, log_probs, actions, rewards, dones from the memory."""
    memory_np: np.ndarray = np.array(memory)

    states = torch.from_numpy(np.vstack(memory_np[:, 0])).float().to(device)
    log_probs = torch.from_numpy(np.vstack(memory_np[:, 1])).float().to(device)
    actions = torch.from_numpy(np.vstack(memory_np[:, 2])).float().to(device)
    rewards = torch.from_numpy(np.vstack(memory_np[:, 3])).float().to(device)
    dones = (
        torch.from_numpy(np.vstack(memory_np[:, 4]).astype(np.uint8)).float().to(device)
    )

    return states, log_probs, actions, rewards, dones


def ppo_iter(
    epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    log_probs: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Yield mini-batches."""
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], log_probs[rand_ids, :], actions[
                rand_ids, :
            ], returns[rand_ids, :], advantages[rand_ids, :]
