import argparse
from typing import Tuple

import torch

from rl_algorithms.common.abstract.learner import TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.registry import LEARNERS
from rl_algorithms.sac.learner import SACLearner
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class SACfDLearner(SACLearner):
    """Learner for BCSAC Agent.

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
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
        SACLearner.__init__(
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

        self.use_n_step = self.hyper_params.n_step > 1

    # pylint: disable=too-many-statements
    def update_model(self, experience: Tuple[TensorTuple, ...]) -> TensorTuple:
        if self.use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        states, actions, rewards, next_states, dones = experience_1[:-3]
        weights, indices, eps_d = experience_1[-3:]
        new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)

        # train alpha
        if self.hyper_params.auto_entropy_tuning:
            alpha_loss = torch.mean(
                (-self.log_alpha * (log_prob + self.target_entropy).detach()) * weights
            )

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params.w_entropy

        # Q function loss
        masks = 1 - dones
        gamma = self.hyper_params.gamma
        states_actions = torch.cat((states, actions), dim=-1)
        q_1_pred = self.qf_1(states_actions)
        q_2_pred = self.qf_2(states_actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.hyper_params.gamma * v_target * masks
        qf_1_loss = torch.mean((q_1_pred - q_target.detach()).pow(2) * weights)
        qf_2_loss = torch.mean((q_2_pred - q_target.detach()).pow(2) * weights)

        if self.use_n_step:
            _, _, rewards, next_states, dones = experience_n

            gamma = gamma ** self.hyper_params.n_step
            masks = 1 - dones

            v_target = self.vf_target(next_states)
            q_target = rewards + gamma * v_target * masks
            qf_1_loss_n = torch.mean((q_1_pred - q_target.detach()).pow(2) * weights)
            qf_2_loss_n = torch.mean((q_2_pred - q_target.detach()).pow(2) * weights)

            # to update loss and priorities
            qf_1_loss = qf_1_loss + qf_1_loss_n * self.hyper_params.lambda1
            qf_2_loss = qf_2_loss + qf_2_loss_n * self.hyper_params.lambda1

        # V function loss
        states_actions = torch.cat((states, new_actions), dim=-1)
        v_pred = self.vf(states)
        q_pred = torch.min(self.qf_1(states_actions), self.qf_2(states_actions))
        v_target = (q_pred - alpha * log_prob).detach()
        vf_loss_element_wise = (v_pred - v_target).pow(2)
        vf_loss = torch.mean(vf_loss_element_wise * weights)

        # train Q functions
        self.qf_1_optim.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optim.step()

        self.qf_2_optim.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optim.step()

        # train V function
        self.vf_optim.zero_grad()
        vf_loss.backward()
        self.vf_optim.step()

        # actor loss
        advantage = q_pred - v_pred.detach()
        actor_loss_element_wise = alpha * log_prob - advantage
        actor_loss = torch.mean(actor_loss_element_wise * weights)

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
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update target networks
        common_utils.soft_update(self.vf, self.vf_target, self.hyper_params.tau)

        # update priorities
        new_priorities = vf_loss_element_wise
        new_priorities += self.hyper_params.lambda3 * actor_loss_element_wise.pow(2)
        new_priorities += self.hyper_params.per_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        new_priorities += eps_d

        return (
            actor_loss.item(),
            qf_1_loss.item(),
            qf_2_loss.item(),
            vf_loss.item(),
            alpha_loss.item(),
            indices,
            new_priorities,
        )
