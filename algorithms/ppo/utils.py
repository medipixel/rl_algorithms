# -*- coding: utf-8 -*-
"""Utility functions for PPO.

This module has PPO util functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

import numpy as np
import torch


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    """Compute gae."""
    values = values + [next_value]
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])

    return returns


def ppo_iter(
    epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Yield mini-batches."""
    batch_size = states.size(0)
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], values[
                rand_ids, :
            ], log_probs[rand_ids, :], returns[rand_ids, :], advantages[rand_ids, :]
