import argparse
from typing import Tuple, Union

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from rl_algorithms.common.abstract.learner import TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.dqn.learner import DQNLearner
from rl_algorithms.registry import LEARNERS
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class DQfDLearner(DQNLearner):
    """Learner for DDPGfD Agent.

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        optim_cfg (ConfigDict): config of optimizer
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optim (Optimizer): optimizer for training actor
        critic_optim (Optimizer): optimizer for training critic

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
        device: torch.device,
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
            device,
        )

    def update_model(
        self, experience: Union[TensorTuple, Tuple[TensorTuple]]
    ) -> TensorTuple:  # type: ignore
        """Train the model after each episode."""

        if self.use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        weights, indices, eps_d = experience_1[-3:]
        actions = experience_1[1]

        # 1 step loss
        gamma = self.hyper_params.gamma
        dq_loss_element_wise, q_values = self.loss_fn(
            self.dqn, self.dqn_target, experience_1, gamma, self.head_cfg
        )
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self.loss_fn(
                self.dqn, self.dqn_target, experience_n, gamma, self.head_cfg
            )

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.lambda1
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # supervised loss using demo for only demo transitions
        demo_idxs = np.where(eps_d != 0.0)
        n_demo = demo_idxs[0].size
        if n_demo != 0:  # if 1 or more demos are sampled
            # get margin for each demo transition
            action_idxs = actions[demo_idxs].long()
            margin = torch.ones(q_values.size()) * self.hyper_params.margin
            margin[demo_idxs, action_idxs] = 0.0  # demo actions have 0 margins
            margin = margin.to(self.device)

            # calculate supervised loss
            demo_q_values = q_values[demo_idxs, action_idxs].squeeze()
            supervised_loss = torch.max(q_values + margin, dim=-1)[0]
            supervised_loss = supervised_loss[demo_idxs] - demo_q_values
            supervised_loss = torch.mean(supervised_loss) * self.hyper_params.lambda2
        else:  # no demo sampled
            supervised_loss = torch.zeros(1, device=self.device)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params.w_q_reg

        # total loss
        loss = dq_loss + supervised_loss + q_regular

        # train dqn
        self.dqn_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy().squeeze()
        new_priorities = loss_for_prior + self.hyper_params.per_eps
        new_priorities += eps_d

        if self.head_cfg.configs.use_noisy_net:
            self.dqn.head.reset_noise()
            self.dqn_target.head.reset_noise()

        return (
            loss.item(),
            dq_loss.item(),
            supervised_loss.item(),
            q_values.mean().item(),
            n_demo,
            indices,
            new_priorities,
        )
