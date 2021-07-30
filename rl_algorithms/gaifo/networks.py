# -*- coding: utf-8 -*-

from typing import Tuple, Union

import torch
import torch.nn as nn

from rl_algorithms.common.helper_functions import identity
from rl_algorithms.registry import build_backbone, build_head
from rl_algorithms.utils.config import ConfigDict


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
class Discriminator(nn.Module):
    """Discriminator to classify experience data and expert data"""

    def __init__(
        self,
        backbone_cfg: ConfigDict,
        head_cfg: ConfigDict,
        shared_backbone: nn.Module = None,
    ):
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

        head_cfg.configs.input_size *= 2

        self.head = build_head(head_cfg)

    def forward(
        self, state_action: Tuple[torch.Tensor, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward method implementation. Use in get_action method in agent."""
        state_feature = self.backbone(state_action[0])
        action_feature = state_action[1]
        return self.head(torch.cat([state_feature, action_feature], dim=-1))

    def calculate_fc_input_size(self, state_dim: tuple):
        """Calculate fc input size according to the shape of cnn."""
        x = torch.zeros(state_dim).unsqueeze(0)
        output = self.backbone(x).detach().view(-1)
        return output.shape[0]
