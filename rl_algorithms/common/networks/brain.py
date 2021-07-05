# -*- coding: utf-8 -*-
"""Brain module for backbone & head holder.

- Authors: Euijin Jeong, Kyunghwan Kim
- Contacts: euijin.jeong@medipixel.io
            kh.kim@medipixel.io
"""

from typing import Tuple, Union

import torch
import torch.nn as nn

from rl_algorithms.common.helper_functions import identity
from rl_algorithms.dqn.networks import IQNMLP
from rl_algorithms.recurrent.utils import infer_leading_dims, restore_leading_dims
from rl_algorithms.registry import build_backbone, build_head
from rl_algorithms.utils.config import ConfigDict


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
class Brain(nn.Module):
    """Class for holding backbone and head networks."""

    def __init__(
        self,
        backbone_cfg: ConfigDict,
        head_cfg: ConfigDict,
        shared_backbone: nn.Module = None,
    ):
        """Initialize."""
        nn.Module.__init__(self)
        if shared_backbone is not None:
            self.backbone = shared_backbone
            head_cfg.configs.input_size = self.calculate_fc_input_size(
                head_cfg.configs.state_size
            )
        elif not backbone_cfg:
            self.backbone = identity
            head_cfg.configs.input_size = head_cfg.configs.state_size[0]
        else:
            self.backbone = build_backbone(backbone_cfg)
            head_cfg.configs.input_size = self.calculate_fc_input_size(
                head_cfg.configs.state_size
            )
        self.head = build_head(head_cfg)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
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


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
class GRUBrain(Brain):
    """Class for holding backbone, GRU, and head networks."""

    def __init__(
        self,
        backbone_cfg: ConfigDict,
        head_cfg: ConfigDict,
        gru_cfg: ConfigDict,
    ):
        self.action_size = head_cfg.configs.output_size
        """Initialize. Generate different structure whether it has CNN module or not."""
        Brain.__init__(self, backbone_cfg, head_cfg)
        self.fc = nn.Linear(
            head_cfg.configs.input_size,
            gru_cfg.rnn_hidden_size,
        )
        self.gru = nn.GRU(
            gru_cfg.rnn_hidden_size + self.action_size + 1,  # 1 is for prev_reward
            gru_cfg.rnn_hidden_size,
            batch_first=True,
        )

        head_cfg.configs.input_size = gru_cfg.rnn_hidden_size
        self.head = build_head(head_cfg)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
    ):
        """Forward method implementation. Use in get_action method in agent.
        We adopted RL^2 algorithm's architecture, which recieve
        previous action & rewrad as RNN input.

        RL^2 paper : https://arxiv.org/pdf/1611.02779.pdf

        Args:
            x (torch.Tensor): state of the transition
            hidden (torch.Tensor): hidden_state of the transition
            prev_action (torch.Tensor): previous transition's action
            prev_reward (torch.Tensor): previous transition's reward
        """

        # Pre-process input for backbone and get backbone's output
        if isinstance(self.backbone, nn.Module):
            lead_dim, batch_len, seq_len, x_shape = infer_leading_dims(x, 3)

            backbone_out = self.backbone(
                x.contiguous().view(batch_len * seq_len, *x_shape)
            )  # Fold if T dimension.
        else:
            if len(x.shape) == 1:
                x = x.reshape(1, 1, -1)
            lead_dim, batch_len, seq_len, x_shape = infer_leading_dims(x, 1)
            backbone_out = x

        # Pass fc layer
        gru_input = self.fc(backbone_out)

        # Make gru_input concat with hidden_state, previous_action and previous_reward.
        gru_input = torch.cat(
            [
                gru_input.view(batch_len, seq_len, -1),
                prev_action.view(batch_len, seq_len, -1),
                prev_reward.view(batch_len, seq_len, 1),
            ],
            dim=2,
        )
        hidden = torch.transpose(hidden, 0, 1)

        # Unroll gru
        gru_out, hidden = self.gru(gru_input.float(), hidden.float())

        # Get q
        q = self.head(gru_out.contiguous().view(batch_len * seq_len, -1))

        # Restore leading dimensions
        q = restore_leading_dims(q, lead_dim, batch_len, seq_len)

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

        # Pre-process input for backbone and get backbone's output
        if isinstance(self.backbone, nn.Module):
            lead_dim, batch_len, seq_len, x_shape = infer_leading_dims(x, 3)

            backbone_out = self.backbone(
                x.contiguous().view(batch_len * seq_len, *x_shape)
            )  # Fold if T dimension.
        else:
            if len(x.shape) == 1:
                x = x.reshape(1, 1, -1)
            lead_dim, batch_len, seq_len, x_shape = infer_leading_dims(x, 1)
            backbone_out = x

        # Pass gru layer
        gru_input = self.fc(backbone_out)

        # Make gru_input concat with hidden_state, previous_action and previous_reward.
        gru_input = torch.cat(
            [
                gru_input.view(batch_len, seq_len, -1),
                prev_action.view(batch_len, seq_len, -1),
                prev_reward.view(batch_len, seq_len, 1),
            ],
            dim=2,
        )
        hidden = torch.transpose(hidden, 0, 1)

        # Unroll gru
        gru_out, hidden = self.gru(gru_input, hidden)

        if isinstance(self.head, IQNMLP):
            quantile_values, quantiles = self.head.forward_(
                gru_out.contiguous().view(batch_len * seq_len, -1), n_tau_samples
            )
            # Restore leading dimensions
            quantile_values = restore_leading_dims(
                quantile_values, lead_dim, batch_len * n_tau_samples, seq_len
            )
            quantiles = restore_leading_dims(
                quantiles, lead_dim, batch_len * n_tau_samples, seq_len
            )

            return quantile_values, quantiles, hidden
        else:
            head_out = self.head.forward_(
                gru_out.contiguous().view(batch_len * seq_len, -1)
            )
            if len(head_out) != 2:  # C51 output is not going to be restore.
                head_out = restore_leading_dims(head_out, lead_dim, batch_len, seq_len)
            return head_out, hidden

    def calculate_fc_input_size(self, state_dim: tuple):
        """Calculate fc input size according to the shape of cnn."""
        x = torch.zeros(state_dim).unsqueeze(0)
        output = self.backbone(x).detach().view(-1)
        return output.shape[0]
