# -*- coding: utf-8 -*-
"""ResNet modules for RL algorithms.

- Authors: Minseop Kim & Kyunghwan Kim
- Contacts: minseop.kim@medipixel.io
            kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1512.03385.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_algorithms.registry import BACKBONES
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
class BasicBlock(nn.Module):
    """Basic building block for ResNet."""

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, expansion: int = 1
    ):
        super(BasicBlock, self).__init__()

        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            self.expansion * planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
class Bottleneck(nn.Module):
    """Bottleneck building block."""

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, expansion: int = 1
    ):
        super(Bottleneck, self).__init__()

        self.expansion = expansion
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# TODO: Remove it when upgrade torch>=1.7
# pylint: disable=abstract-method
@BACKBONES.register_module
class ResNet(nn.Module):
    """Baseline of ResNet(https://arxiv.org/pdf/1512.03385.pdf)."""

    def __init__(
        self,
        configs: ConfigDict,
    ):
        super(ResNet, self).__init__()
        block = Bottleneck if configs.use_bottleneck else BasicBlock
        block_outputs = configs.block_output_sizes
        block_strides = configs.block_strides
        num_blocks = configs.num_blocks
        self.expansion = configs.expansion
        self.in_planes = configs.first_output_size
        self.conv1 = nn.Conv2d(
            configs.first_input_size,
            self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(
            block, block_outputs[0], num_blocks[0], block_strides[0]
        )
        self.layer2 = self._make_layer(
            block, block_outputs[1], num_blocks[1], block_strides[1]
        )
        self.layer3 = self._make_layer(
            block, block_outputs[2], num_blocks[2], block_strides[2]
        )
        self.layer4 = self._make_layer(
            block, block_outputs[3], num_blocks[3], block_strides[3]
        )
        self.conv_out = nn.Conv2d(
            block_outputs[3] * self.expansion,
            block_outputs[3] // configs.channel_compression,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_ in strides:
            layers.append(block(self.in_planes, planes, stride_, self.expansion))
            self.in_planes = planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_out(x)
        x = x.view(x.size(0), -1)
        return x
