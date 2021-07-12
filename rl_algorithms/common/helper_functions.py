# -*- coding: utf-8 -*-
"""Common util functions for all algorithms.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""
from collections import OrderedDict, deque
import random
from typing import Deque, Dict, List, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn

from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tanh(x: torch.Tensor) -> torch.Tensor:
    """Return torch.tanh(x)"""
    return torch.tanh(x)


def identity(x: torch.Tensor) -> torch.Tensor:
    """Return input without any change."""
    return x


def relu(x: torch.Tensor) -> torch.Tensor:
    """Return torch.relu(x)"""
    return torch.relu(x)


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


def numpy2floattensor(
    arrays: Union[np.ndarray, Tuple[np.ndarray]], device_: torch.device
) -> Tuple[torch.Tensor]:
    """Convert numpy type to torch FloatTensor.
    - Convert numpy array to torch float tensor.
    - Convert numpy array with Tuple type to torch FloatTensor with Tuple.
    """

    if isinstance(arrays, tuple):  # check Tuple or not
        tensors = []
        for array in arrays:
            tensor = (
                torch.from_numpy(array.copy()).to(device_, non_blocking=True).float()
            )
            tensors.append(tensor)
        return tuple(tensors)
    tensor = torch.from_numpy(arrays.copy()).to(device_, non_blocking=True).float()
    return tensor


def state_dict2numpy(state_dict) -> Dict[str, np.ndarray]:
    """Convert Pytorch state dict to list of numpy arrays."""
    np_state_dict = OrderedDict()
    for param in list(state_dict):
        np_state_dict[param] = state_dict[param].numpy()
    return np_state_dict


def smoothen_graph(scalars: List[float], weight: float = 0.6) -> List[float]:
    """Smoothen result graph using exponential moving average formula as TensorBoard.

    Reference:
        https://docs.wandb.com/library/technical-faq#what-formula-do-you-use-for-
        your-smoothing-algorithm
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        # Calculate smoothed value
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


def set_cfg_for_intergration_test(cfg: ConfigDict) -> ConfigDict:
    """Set specific values in config for intergration test."""
    if "batch_size" in cfg.agent.hyper_params:
        cfg.agent.hyper_params.batch_size = 10
    if "update_starts_from" in cfg.agent.hyper_params:
        cfg.agent.hyper_params.update_starts_from = 50
    if "initial_random_action" in cfg.agent.hyper_params:
        cfg.agent.hyper_params.initial_random_action = 10
    if cfg.agent.type == "ApeX":
        cfg.agent.hyper_params.num_workers = 1
        cfg.agent.hyper_params.worker_update_interval = 1
        cfg.agent.hyper_params.logger_interval = 1
        cfg.agent.hyper_params.buffer_size = 50
        cfg.agent.hyper_params.max_update_step = 1
        initial_port = random.randint(6000, 8000)
        cfg.agent.comm_cfg.learner_buffer_port = initial_port
        cfg.agent.comm_cfg.learner_worker_port = initial_port + 1
        cfg.agent.comm_cfg.learner_logger_port = initial_port + 2
        cfg.agent.comm_cfg.send_batch_port = initial_port + 3
        cfg.agent.comm_cfg.priorities_port = initial_port + 4
    if cfg.agent.type == "PPOAgent":
        cfg.agent.hyper_params.epoch = 1
        cfg.agent.hyper_params.n_workers = 1
        cfg.agent.hyper_params.rollout_len = 10
    if "fD" in cfg.agent.type:
        cfg.agent.hyper_params.pretrain_step = 1
    if "Distillation" in cfg.agent.type:
        cfg.agent.hyper_params.n_frame_from_last = 3
        cfg.agent.hyper_params.save_dir = "data/"
    return cfg
