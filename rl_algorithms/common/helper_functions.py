# -*- coding: utf-8 -*-
"""Common util functions for all algorithms.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from collections import deque
import random
from typing import Deque, List, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input without any change."""
    return x


def soft_update(local: nn.Module, target: nn.Module, tau: float):
    """Soft-update: target = tau*local + (1-tau)*target."""
    for t_param, l_param in zip(target.parameters(), local.parameters()):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


def hard_update(local: nn.Module, target: nn.Module):
    """Hard update: target <- local."""
    target.load_state_dict(local.state_dict())


def set_random_seed(seed: int, env: gym.Env):
    """Set random seed"""
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_one_hot(labels: torch.Tensor, c: int):
    """Converts an integer label to a one-hot Variable."""
    y = torch.eye(c).to(device)
    labels = labels.type(torch.LongTensor)
    return y[labels]


def get_n_step_info_from_demo(
    demo: List, n_step: int, gamma: float
) -> Tuple[List, List]:
    """Return 1 step and n step demos."""
    assert n_step > 1

    demos_1_step = list()
    demos_n_step = list()
    n_step_buffer: Deque = deque(maxlen=n_step)

    for transition in demo:
        n_step_buffer.append(transition)

        if len(n_step_buffer) == n_step:
            # add a single step transition
            demos_1_step.append(n_step_buffer[0])

            # add a multi step transition
            curr_state, action = n_step_buffer[0][:2]
            reward, next_state, done = get_n_step_info(n_step_buffer, gamma)
            transition = (curr_state, action, reward, next_state, done)
            demos_n_step.append(transition)

    return demos_1_step, demos_n_step


def get_n_step_info(
    n_step_buffer: Deque, gamma: float
) -> Tuple[np.int64, np.ndarray, bool]:
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]

        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done


def infer_leading_dims(tensor, dim):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.

    Cloned at rlpyt repo:
    https://github.com/astooke/rlpyt/blob/master/rlpyt/models/dqn/atari_r2d1_model.py
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


def restore_leading_dims(tensors, lead_dim, T=1, B=1):
    """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
    leading dimensions, which will become [], [B], or [T,B].  Assumes input
    tensors already have a leading Batch dimension, which might need to be
    removed. (Typically the last layer of model will compute with leading
    batch dimension.)  For use in model ``forward()`` method, so that output
    dimensions match input dimensions, and the same model can be used for any
    such case.  Use with outputs from ``infer_leading_dims()``."""
    is_seq = isinstance(tensors, (tuple, list))
    tensors = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put T dim.)
        tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
    if lead_dim == 0:  # (Remove B=1 dim.)
        assert B == 1
        tensors = tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]


def valid_from_done(done):
    """Returns a float mask which is zero for all time-steps after a
    `done=True` is signaled.  This function operates on the leading dimension
    of `done`, assumed to correspond to time [T,...], other dimensions are
    preserved.
    Cloned at rlpyt repo:
        https://github.com/astooke/rlpyt/blob/master/rlpyt/algos/utils.py
    """
    done = done.type(torch.float).squeeze()
    valid = torch.ones_like(done)
    valid[:, 1:] = 1 - torch.clamp(torch.cumsum(done[:, :-1], dim=0), max=1)
    valid = valid[:, -1] == 0
    valid = valid.unsqueeze(-1)
    return valid
