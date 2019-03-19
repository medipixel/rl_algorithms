# -*- coding: utf-8 -*-
"""Utility functions for DQN.

This module has DQN util functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from algorithms.dqn.networks import CategoricalDuelingMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_dqn_c51_loss(
    model: CategoricalDuelingMLP,
    target_model: CategoricalDuelingMLP,
    experiences: Tuple[torch.Tensor, ...],
    gamma: float,
    batch_size: int,
    v_min: int,
    v_max: int,
    atom_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return element-wise dqn loss and Q-values."""
    states, actions, rewards, next_states, dones = experiences[:5]
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

    dist, q_values = model.get_dist_q(states)
    log_p = torch.log(dist[range(batch_size), actions.long()])

    dq_loss_element_wise = -(proj_dist * log_p).sum(1)

    return dq_loss_element_wise, q_values


def get_dqn_loss(
    model: CategoricalDuelingMLP,
    target_model: CategoricalDuelingMLP,
    experiences: Tuple[torch.Tensor, ...],
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return element-wise dqn loss and Q-values."""
    states, actions, rewards, next_states, dones = experiences[:5]

    q_values = model(states)
    next_q_values = model(next_states)
    next_target_q_values = target_model(next_states)

    curr_q_value = q_values.gather(1, actions.long().unsqueeze(1))
    next_q_value = next_target_q_values.gather(  # Double DQN
        1, next_q_values.argmax(1).unsqueeze(1)
    )

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
    #       = r                       otherwise
    masks = 1 - dones
    target = rewards + gamma * next_q_value * masks
    target = target.to(device)

    # calculate dq loss
    dq_loss_element_wise = F.smooth_l1_loss(
        curr_q_value, target.detach(), reduction="none"
    )

    return dq_loss_element_wise, q_values
