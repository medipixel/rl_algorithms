import argparse
from typing import Callable, Tuple, Union

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class DQNLearner(Learner):
    """Learner for DQN Agent

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters

    """

    def __init__(
        self,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        head_cfg: ConfigDict,
        loss_fn: Callable,
        device: torch.device,
    ):
        Learner.__init__(self, args, hyper_params, device)
        self.head_cfg = head_cfg
        self.loss_fn = loss_fn
        self.use_n_step = hyper_params.n_step > 1

    def update_model(
        self,
        networks: Tuple[Brain, ...],
        optimizer: Union[optim.Optimizer, Tuple[optim.Optimizer, ...]],
        experience: Union[TensorTuple, Tuple[TensorTuple]],
    ) -> Tuple[torch.Tensor, torch.Tensor, list, np.ndarray]:  # type: ignore
        """Update dqn and dqn target"""
        dqn, dqn_target = networks
        dqn_optim = optimizer

        if self.use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        weights, indices = experience_1[-3:-1]
        gamma = self.hyper_params.gamma
        dq_loss_element_wise, q_values = self.loss_fn(
            dqn, dqn_target, experience_1, gamma, self.head_cfg
        )
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        if self.use_n_step:
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self.loss_fn(
                dqn, dqn_target, experience_n, gamma, self.head_cfg
            )

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params.w_q_reg

        # total loss
        loss = dq_loss + q_regular

        dqn_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(dqn.parameters(), self.hyper_params.gradient_clip)
        dqn_optim.step()

        # update target networks
        common_utils.soft_update(dqn, dqn_target, self.hyper_params.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params.per_eps

        if self.head_cfg.configs.use_noisy_net:
            dqn.head.reset_noise()
            dqn_target.head.reset_noise()

        return loss.item(), q_values.mean().item(), indices, new_priorities
