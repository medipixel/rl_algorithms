import torch
import torch.nn as nn

from rl_algorithms.common.helper_functions import identity
from rl_algorithms.common.networks.cnn import CNN
from rl_algorithms.common.networks.mlp import FlattenMLP
from rl_algorithms.dqn.networks import IQNMLP
from rl_algorithms.registry import build_backbone, build_head
from rl_algorithms.utils.config import ConfigDict


class BaseNetwork(nn.Module):
    """this class is for holding backbone and head networks"""

    def __init__(
        self, backbone_cfg: ConfigDict, head_cfg: ConfigDict,
    ):
        super(BaseNetwork, self).__init__()
        if not backbone_cfg:
            self.backbone = identity
            head_cfg.configs.input_size = head_cfg.configs.state_size[0]
        else:
            self.backbone = build_backbone(backbone_cfg)
            head_cfg.configs.input_size = calculate_fc_input_size(
                head_cfg.configs.state_size, backbone_cfg
            )
        self.head = build_head(head_cfg)

    def forward(self, x, n_tau_samples: int = None, actions: torch.Tensor = None):
        """use in get_action method in agent"""
        x = self.backbone(x)
        if isinstance(self.head, IQNMLP):
            x = self.head.forward(x, n_tau_samples)
        elif isinstance(self.head, FlattenMLP):
            x = self.head.forward(x, actions)
        else:
            x = self.head.forward(x)
        return x

    def forward_(self, x: torch.Tensor, n_tau_samples: int = None):
        x = self.backbone(x)
        if isinstance(self.head, IQNMLP):
            x = self.head.forward_(x, n_tau_samples)
        else:
            x = self.head.forward_(x)
        return x


def calculate_fc_input_size(state_dim: tuple, cnn: ConfigDict):
    """calculate fc input size according to the shape of cnn"""
    x = torch.zeros(state_dim).unsqueeze(0)

    cnn = cnn
    cnn_model = CNN(cnn.configs)
    cnn_output = cnn_model.get_cnn_features(x).view(-1)
    return cnn_output.shape[0]
