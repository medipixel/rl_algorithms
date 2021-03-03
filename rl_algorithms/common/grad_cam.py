"""Grad-CAM class for analyzing CNN network.

- Author: Kyunghwan Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1610.02391v1.pdf
- Reference: https://github.com/RRoundTable/XAI
"""

from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# pylint: disable=attribute-defined-outside-init
class CAMBaseWrapper:
    """Base Wrapping module for CAM."""

    def __init__(self, model: nn.Module):
        """Initialize."""
        super(CAMBaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids: torch.Tensor) -> torch.Tensor:
        """Convert input to one-hot."""
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot[0][ids] = 1
        return one_hot

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Simple classification
        """
        self.model.zero_grad()
        self.logits = self.model(image)
        return self.logits

    def backward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Class-specific backpropagation.
        Either way works:
        1. self.logits.backward(gradient=one_hot, retain_graph=True)
        2. (self.logits * one_hot).sum().backward(retain_graph=True)
        """

        one_hot = self._encode_one_hot(ids)
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self, target_layer: str):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


# pylint: disable=attribute-defined-outside-init
class GradCAM(CAMBaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model: nn.Module, candidate_layers: list = None):
        """Initialize."""
        super(GradCAM, self).__init__(model)
        self.fmap_pool = OrderedDict()
        self.grad_pool = OrderedDict()
        self.candidate_layers = candidate_layers  # list

        def forward_hook(key: str) -> Callable:
            def forward_hook_(_, __, output: torch.Tensor):
                # Save featuremaps
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key: str) -> Callable:
            def backward_hook_(_, __, grad_out: tuple):
                # Save the gradients correspond to the featuremaps
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            print(name, module)
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    @staticmethod
    def _find(pool: OrderedDict, target_layer: str) -> torch.Tensor:
        """Get designated layer from model."""
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    @staticmethod
    def _compute_grad_weights(grads: torch.Tensor) -> torch.Tensor:
        """Compute gradient weight with average pooling."""
        return F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image: np.ndarray) -> torch.Tensor:
        """Forward method implementation."""
        self.image_shape = image.shape[1:]
        return super(GradCAM, self).forward(image)

    def generate(self, target_layer: str) -> torch.Tensor:
        """Generate feature map of target layer with Grad-CAM."""
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0] + 1e-7
        gcam = gcam.view(B, C, H, W)

        return gcam
