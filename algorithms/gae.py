# -*- coding: utf-8 -*-
"""High-Dimensional Continuous Control Using Generalized Advantage Estimation.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1506.02438.pdf

"""

import torch


class GAE:
    """Return returns and advantages.

    Example:
        gae = GAE()
        returns, advantages = gae.get_gae(
            rewards, values, dones, gamma=0.99, labd=0.95
        )

    """

    @staticmethod
    def get_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        lambd: float,
        normalize: bool = True,
    ):
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

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return returns, advantages
