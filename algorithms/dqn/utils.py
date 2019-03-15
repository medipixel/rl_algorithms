# -*- coding: utf-8 -*-
"""Utility functions for DQN.

This module has DQN util functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import torch

from algorithms.dqn.networks import CategoricalDuelingMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def projection_distribution(
    target_model: CategoricalDuelingMLP,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    v_min: int,
    v_max: int,
    atom_size: int,
    gamma: float,
) -> torch.Tensor:
    """Get projection distribution (C51) to calculate dqn loss.

    This code taken and modified from:
        https://github.com/higgsfield/RL-Adventure
    """
    batch_size = next_states.size(0)
    delta_z = float(v_max - v_min) / (atom_size - 1)

    support = torch.linspace(v_min, v_max, atom_size).to(device)
    next_dist, q = target_model.get_dist_q(next_states)
    next_actions = q.argmax(1)
    next_actions = (
        next_actions.unsqueeze(1)
        .unsqueeze(1)
        .expand(next_dist.size(0), 1, next_dist.size(2))
    )
    next_dist = next_dist.gather(1, next_actions).squeeze(1)

    t_z = rewards + (1 - dones) * gamma * support
    t_z = t_z.clamp(min=v_min, max=v_max)
    b = (t_z - v_min) / delta_z
    l = b.floor().long()  # noqa: E741
    u = b.ceil().long()

    offset = (
        torch.linspace(0, (batch_size - 1) * atom_size, batch_size)
        .long()
        .unsqueeze(1)
        .expand(batch_size, atom_size)
        .to(device)
    )

    proj_dist = torch.zeros(next_dist.size(), device=device)
    index = (l + offset).view(-1)
    proj_dist.view(-1).index_add_(0, index, (next_dist * (u.float() - b)).view(-1))
    index = (u + offset).view(-1)
    proj_dist.view(-1).index_add_(0, index, (next_dist * (b - l.float())).view(-1))

    return proj_dist
