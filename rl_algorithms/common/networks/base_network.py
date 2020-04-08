import torch
import torch.nn as nn

from rl_algorithms.dqn.networks import IQNMLP

class Base_network(nn.Module):
    def __init__(
        self,
        backbone,
        head,
    ):
        super(Base_network, self).__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x, n_tau_samples: int = None):
        x = self.backbone(x)
        if isinstance(self.head, IQNMLP) :
            x = self.head.forward(x, n_tau_samples)
        else :
            x = self.head.forward(x)
        return x
    
    def forward_(self, x: torch.Tensor, n_tau_samples: int = None):
        x = self.backbone(x)
        if isinstance(self.head, IQNMLP) :
            x = self.head.forward_(x, n_tau_samples)
        else :
            x = self.head.forward_(x)
        return x