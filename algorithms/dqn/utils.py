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
    model: CategoricalDuelingMLP,
    target_model: CategoricalDuelingMLP,
    batch_size: int,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    v_min: int,
    v_max: int,
    atom_size: int,
    gamma: float,
) -> torch.Tensor:
    """Get projection distribution (C51) to calculate dqn loss.  """
    support = torch.linspace(v_min, v_max, atom_size).to(device)
    delta_z = float(v_max - v_min) / (atom_size - 1)

    with torch.no_grad():
        next_actions = model.get_dist_q(next_states)[1].argmax(1)
        next_dist = target_model.get_dist_q(next_states)[0]
        next_dist = next_dist[range(batch_size), next_actions]

        t_z = rewards + (1 - dones) * gamma * support
        t_z = t_z.clamp(min=v_min, max=v_max)
        b = (t_z - v_min) / delta_z
        l = b.floor().long()  # noqa: E741
        u = b.ceil().long()

        # Fix disappearing probability mass when l = b = u (b is int)
        # taken from https://github.com/Kaixhin/Rainbow
        l[(u > 0) * (l == u)] -= 1  # noqa: E741
        u[(l < (atom_size - 1)) * (l == u)] += 1  # noqa: E741

        offset = (
            torch.linspace(0, (batch_size - 1) * atom_size, batch_size)
            .long()
            .unsqueeze(1)
            .expand(batch_size, atom_size)
            .to(device)
        )

        proj_dist = torch.zeros(next_dist.size(), device=device)
        proj_dist.view(-1).index_add_(
            0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
        )
        proj_dist.view(-1).index_add_(
            0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
        )

    return proj_dist
