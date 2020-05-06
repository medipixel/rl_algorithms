# -*- coding: utf-8 -*-
"""Brain module for backbone & head holder.

- Authors: Euijin Jeong & Kyunghwan Kim
- Contacts: euijin.jeong@medipixel.io
            kh.kim@medipixel.io
"""

import torch
import torch.nn as nn

from rl_algorithms.common.helper_functions import (
    identity,
    infer_leading_dims,
    restore_leading_dims,
)
from rl_algorithms.dqn.networks import IQNMLP
from rl_algorithms.registry import build_backbone, build_head
from rl_algorithms.utils.config import ConfigDict


class Brain(nn.Module):
    """Class for holding backbone and head networks."""

    def __init__(
        self, backbone_cfg: ConfigDict, head_cfg: ConfigDict,
    ):
        """Initialize."""
        super(Brain, self).__init__()
        if not backbone_cfg:
            self.backbone = identity
            head_cfg.configs.input_size = head_cfg.configs.state_size[0]
        else:
            self.backbone = build_backbone(backbone_cfg)
            head_cfg.configs.input_size = self.calculate_fc_input_size(
                head_cfg.configs.state_size
            )
        self.head = build_head(head_cfg)

    def forward(self, x: torch.Tensor):
        """Forward method implementation. Use in get_action method in agent."""
        x = self.backbone(x)
        x = self.head(x)

        return x

    def forward_(self, x: torch.Tensor, n_tau_samples: int = None):
        """Get output value for calculating loss."""
        x = self.backbone(x)
        if isinstance(self.head, IQNMLP):
            x = self.head.forward_(x, n_tau_samples)
        else:
            x = self.head.forward_(x)
        return x

    def calculate_fc_input_size(self, state_dim: tuple):
        """Calculate fc input size according to the shape of cnn."""
        x = torch.zeros(state_dim).unsqueeze(0)
        output = self.backbone(x).detach().view(-1)
        return output.shape[0]


class GRUBrain(Brain):
    """Class for holding backbone, GRU, and head networks."""

    def __init__(
        self, backbone_cfg: ConfigDict, head_cfg: ConfigDict,
    ):
        self.action_size = head_cfg.configs.output_size
        """Initialize. Generate different structure whether it has CNN module or not."""
        super(GRUBrain, self).__init__(backbone_cfg, head_cfg)
        if not backbone_cfg:
            self.backbone = identity
            head_cfg.configs.input_size = head_cfg.configs.state_size[0]
            self.fc = nn.Linear(
                head_cfg.configs.input_size, head_cfg.configs.rnn_hidden_size,
            )
            self.gru = nn.GRU(
                head_cfg.configs.rnn_hidden_size
                + self.action_size
                + 1,  # 1 is for prev_reward
                head_cfg.configs.rnn_hidden_size,
                batch_first=True,
            )
        else:
            self.backbone = build_backbone(backbone_cfg)
            head_cfg.configs.input_size = self.calculate_fc_input_size(
                head_cfg.configs.state_size
            )
            self.fc = nn.Linear(
                head_cfg.configs.input_size, head_cfg.configs.rnn_hidden_size,
            )
            self.gru = nn.GRU(
                head_cfg.configs.rnn_hidden_size
                + self.action_size
                + 1,  # 1 is for prev_reward
                head_cfg.configs.rnn_hidden_size,
                batch_first=True,
            )

        head_cfg.configs.input_size = head_cfg.configs.rnn_hidden_size
        self.head = build_head(head_cfg)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
    ):
        """Forward method implementation. Use in get_action method in agent."""
        if isinstance(self.backbone, nn.Module):
            x = x / 255.0
            lead_dim, T, B, x_shape = infer_leading_dims(x, 3)

            backbone_out = self.backbone(
                x.view(T * B, *x_shape)
            )  # Fold if T dimension.
        else:
            if len(x.shape) == 1:
                x = x.reshape(1, 1, -1)
            lead_dim, T, B, x_shape = infer_leading_dims(x, 1)
            backbone_out = x
        lstm_input = self.fc(backbone_out)
        lstm_input = torch.cat(
            [
                lstm_input.view(T, B, -1),
                prev_action.view(T, B, -1),
                prev_reward.view(T, B, 1),
            ],
            dim=2,
        )
        hidden = torch.transpose(hidden, 0, 1)
        hidden = None if hidden is None else hidden
        lstm_out, hidden = self.gru(lstm_input, hidden)

        q = self.head(lstm_out.contiguous().view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)

        return q, hidden

    def forward_(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        n_tau_samples: int = None,
    ):
        """Get output value for calculating loss."""
        if isinstance(self.backbone, nn.Module):
            x = x / 255.0
            lead_dim, T, B, x_shape = infer_leading_dims(x, 3)

            backbone_out = self.backbone(
                x.view(T * B, *x_shape)
            )  # Fold if T dimension.
        else:
            if len(x.shape) == 1:
                x = x.reshape(1, 1, -1)
            lead_dim, T, B, x_shape = infer_leading_dims(x, 1)
            backbone_out = x
        lstm_input = self.fc(backbone_out)
        lstm_input = torch.cat(
            [
                lstm_input.view(T, B, -1),
                prev_action.view(T, B, -1),
                prev_reward.view(T, B, 1),
            ],
            dim=2,
        )
        hidden = torch.transpose(hidden, 0, 1)
        hidden = None if hidden is None else hidden
        lstm_out, hidden = self.gru(lstm_input, hidden)

        if isinstance(self.head, IQNMLP):
            quantile_values, quantiles = self.head.forward_(
                lstm_out.contiguous().view(T * B, -1), n_tau_samples
            )
            # Restore leading dimensions: [T,B], [B], or [], as input.
            quantile_values = restore_leading_dims(
                quantile_values, lead_dim, T * n_tau_samples, B
            )
            quantiles = restore_leading_dims(quantiles, lead_dim, T * n_tau_samples, B)

            return quantile_values, quantiles, hidden
        else:
            head_out = self.head.forward_(lstm_out.contiguous().view(T * B, -1))
            if len(head_out) != 2:  # c51의 head_out은 길이가 2인 튜플이기 때문에 c51을 제외하는 조건문.
                head_out = restore_leading_dims(head_out, lead_dim, T, B)
            return head_out, hidden

    def calculate_fc_input_size(self, state_dim: tuple):
        """Calculate fc input size according to the shape of cnn."""
        x = torch.zeros(state_dim).unsqueeze(0)
        output = self.backbone(x).detach().view(-1)
        return output.shape[0]
