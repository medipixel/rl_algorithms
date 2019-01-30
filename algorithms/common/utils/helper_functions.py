# -*- coding: utf-8 -*-
"""Common util functions for all algorithms.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import torch.nn as nn


def soft_update(local: nn.Module, target: nn.Module, tau: float):
    """Soft-update: target = tau*local + (1-tau)*target."""
    for t_param, l_param in zip(target.parameters(), local.parameters()):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
