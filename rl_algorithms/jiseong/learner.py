"""Learner for DQN Agent.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""
import argparse
from typing import Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.dqn.learner import DQNLearner
from rl_algorithms.registry import LEARNERS, build_loss_js
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class JSLearner(DQNLearner):
    """Learner for DQN Agent.

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        dqn (nn.Module): dqn model to predict state Q values
        dqn_target (nn.Module): target dqn model to predict state Q values
        dqn_optim (Optimizer): optimizer for training dqn

    """

    def __init__(
        self,
        args: argparse.Namespace,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        log_cfg: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
        optim_cfg: ConfigDict,
        loss_type: ConfigDict,
    ):
        DQNLearner.__init__(
            self,
            args,
            env_info,
            hyper_params,
            log_cfg,
            backbone,
            head,
            optim_cfg,
            loss_type,
        )

        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        self.dqn = Brain(self.backbone_cfg, self.head_cfg).to(self.device)
        self.dqn_target = Brain(self.backbone_cfg, self.head_cfg).to(self.device)
        self.loss_fn = build_loss_js(self.loss_type)

        self.dqn_target.load_state_dict(self.dqn.state_dict())

        # create optimizer
        self.dqn_optim = optim.Adam(
            self.dqn.parameters(),
            lr=self.optim_cfg.lr_dqn,
            weight_decay=self.optim_cfg.weight_decay,
            eps=self.optim_cfg.adam_eps,
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def update_model(
        self, experience
    ) -> Tuple[torch.Tensor, torch.Tensor, list, np.ndarray]:
        """Update dqn and dqn target."""

        weights, indices = experience[-3:-1]

        gamma = self.hyper_params.gamma

        dq_loss_element_wise, q_values = self.loss_fn(
            self.dqn, self.dqn_target, experience, gamma, self.head_cfg
        )

        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params.w_q_reg

        # total loss
        loss = dq_loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params.per_eps

        if self.head_cfg.configs.use_noisy_net:
            self.dqn.head.reset_noise()
            self.dqn_target.head.reset_noise()

        return (
            loss.item(),
            q_values.mean().item(),
            indices,
            new_priorities,
        )
