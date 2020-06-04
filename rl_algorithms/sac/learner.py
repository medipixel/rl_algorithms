import argparse
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class SACLearner(Learner):
    """Learner for SAC Agent

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        update_step (int): step number of updates
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optim (Optimizer): optimizer for alpha

    """

    def __init__(
        self, args: argparse.Namespace, hyper_params: ConfigDict, device: torch.device
    ):
        Learner.__init__(self, args, hyper_params, device)

        self.update_step = 0
        if self.hyper_params.auto_entropy_tuning:
            self.target_entropy = -np.prod((self.hyper_params["action_dim"],)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

    def update_model(
        self,
        networks: Tuple[Brain, ...],
        optimizer: Union[optim.Optimizer, Tuple[optim.Optimizer, ...]],
        experience: TensorTuple,
    ) -> TensorTuple:
        """Update ddpg actor and critic networks"""
        actor, vf, vf_target, qf_1, qf_2 = networks
        actor_optim, vf_optim, qf_1_optim, qf_2_optim = optimizer[0:4]
        if self.hyper_params.auto_entropy_tuning:
            alpha_optim = optimizer[-1]

        states, actions, rewards, next_states, dones = experience
        new_actions, log_prob, pre_tanh_value, mu, std = actor(states)

        # train alpha
        if self.hyper_params.auto_entropy_tuning:
            alpha_loss = (
                -self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params.w_entropy

        # Q function loss
        masks = 1 - dones
        states_actions = torch.cat((states, actions), dim=-1)
        q_1_pred = qf_1(states_actions)
        q_2_pred = qf_2(states_actions)
        v_target = vf_target(next_states)
        q_target = rewards + self.hyper_params.gamma * v_target * masks
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        # V function loss
        states_actions = torch.cat((states, new_actions), dim=-1)
        v_pred = vf(states)
        q_pred = torch.min(qf_1(states_actions), qf_2(states_actions))
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        # train Q functions
        qf_1_optim.zero_grad()
        qf_1_loss.backward()
        qf_1_optim.step()

        qf_2_optim.zero_grad()
        qf_2_loss.backward()
        qf_2_optim.step()

        # train V function
        vf_optim.zero_grad()
        vf_loss.backward()
        vf_optim.step()

        if self.update_step % self.hyper_params.policy_update_freq == 0:
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            # regularization
            mean_reg = self.hyper_params.w_mean_reg * mu.pow(2).mean()
            std_reg = self.hyper_params.w_std_reg * std.pow(2).mean()
            pre_activation_reg = self.hyper_params.w_pre_activation_reg * (
                pre_tanh_value.pow(2).sum(dim=-1).mean()
            )
            actor_reg = mean_reg + std_reg + pre_activation_reg

            # actor loss + regularization
            actor_loss += actor_reg

            # train actor
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # update target networks
            common_utils.soft_update(vf, vf_target, self.hyper_params.tau)
        else:
            actor_loss = torch.zeros(1)

        self.update_step = self.update_step + 1

        return (
            actor_loss.item(),
            qf_1_loss.item(),
            qf_2_loss.item(),
            vf_loss.item(),
            alpha_loss.item(),
        )
