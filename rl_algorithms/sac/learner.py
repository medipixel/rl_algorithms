from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.registry import LEARNERS
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class SACLearner(Learner):
    """Learner for SAC Agent.

    Attributes:
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        update_step (int): step number of updates
        target_entropy (int): desired entropy used for the inequality constraint
        log_alpha (torch.Tensor): weight for entropy
        alpha_optim (Optimizer): optimizer for alpha
        actor (nn.Module): actor model to select actions
        actor_optim (Optimizer): optimizer for training actor
        critic_1 (nn.Module): critic model to predict state values
        critic_2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        critic_optim1 (Optimizer): optimizer for training critic_1
        critic_optim2 (Optimizer): optimizer for training critic_2

    """

    def __init__(
        self,
        hyper_params: ConfigDict,
        log_cfg: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
        optim_cfg: ConfigDict,
        env_name: str,
        state_size: tuple,
        output_size: int,
        is_test: bool,
        load_from: str,
    ):
        Learner.__init__(self, hyper_params, log_cfg, env_name, is_test)

        self.backbone_cfg = backbone
        self.head_cfg = head
        self.head_cfg.actor.configs.state_size = (
            self.head_cfg.critic_vf.configs.state_size
        ) = state_size
        self.head_cfg.critic_qf.configs.state_size = (state_size[0] + output_size,)
        self.head_cfg.actor.configs.state_size = state_size
        self.head_cfg.actor.configs.output_size = output_size
        self.optim_cfg = optim_cfg
        self.load_from = load_from

        self.update_step = 0
        if self.hyper_params.auto_entropy_tuning:
            self.target_entropy = -np.prod((output_size,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=optim_cfg.lr_entropy)

        self._init_network()

    # pylint: disable=attribute-defined-outside-init
    def _init_network(self):
        """Initialize networks and optimizers."""
        # create actor
        self.actor = Brain(self.backbone_cfg.actor, self.head_cfg.actor).to(self.device)

        # create v_critic
        self.vf = Brain(self.backbone_cfg.critic_vf, self.head_cfg.critic_vf).to(
            self.device
        )
        self.vf_target = Brain(self.backbone_cfg.critic_vf, self.head_cfg.critic_vf).to(
            self.device
        )
        self.vf_target.load_state_dict(self.vf.state_dict())

        # create q_critic
        self.qf_1 = Brain(self.backbone_cfg.critic_qf, self.head_cfg.critic_qf).to(
            self.device
        )
        self.qf_2 = Brain(self.backbone_cfg.critic_qf, self.head_cfg.critic_qf).to(
            self.device
        )

        # create optimizers
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )
        self.vf_optim = optim.Adam(
            self.vf.parameters(),
            lr=self.optim_cfg.lr_vf,
            weight_decay=self.optim_cfg.weight_decay,
        )
        self.qf_1_optim = optim.Adam(
            self.qf_1.parameters(),
            lr=self.optim_cfg.lr_qf1,
            weight_decay=self.optim_cfg.weight_decay,
        )
        self.qf_2_optim = optim.Adam(
            self.qf_2.parameters(),
            lr=self.optim_cfg.lr_qf2,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # load the optimizer and model parameters
        if self.load_from is not None:
            self.load_params(self.load_from)

    def update_model(
        self, experience: Union[TensorTuple, Tuple[TensorTuple]]
    ) -> Tuple[torch.Tensor, torch.Tensor, list, np.ndarray]:  # type: ignore
        """Update actor and critic networks."""
        self.update_step += 1

        states, actions, rewards, next_states, dones = experience
        new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)

        # train alpha
        if self.hyper_params.auto_entropy_tuning:
            alpha_loss = (
                -self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params.w_entropy

        # Q function loss
        masks = 1 - dones
        states_actions = torch.cat((states, actions), dim=-1)
        q_1_pred = self.qf_1(states_actions)
        q_2_pred = self.qf_2(states_actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.hyper_params.gamma * v_target * masks
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        # V function loss
        states_actions = torch.cat((states, new_actions), dim=-1)
        v_pred = self.vf(states)
        q_pred = torch.min(self.qf_1(states_actions), self.qf_2(states_actions))
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

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
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            common_utils.soft_update(self.vf, self.vf_target, self.hyper_params.tau)
        else:
            actor_loss = torch.zeros(1)

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

        return (
            actor_loss.item(),
            qf_1_loss.item(),
            qf_2_loss.item(),
            vf_loss.item(),
            alpha_loss.item(),
        )

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor": self.actor.state_dict(),
            "qf_1": self.qf_1.state_dict(),
            "qf_2": self.qf_2.state_dict(),
            "vf": self.vf.state_dict(),
            "vf_target": self.vf_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "qf_1_optim": self.qf_1_optim.state_dict(),
            "qf_2_optim": self.qf_2_optim.state_dict(),
            "vf_optim": self.vf_optim.state_dict(),
        }

        if self.hyper_params.auto_entropy_tuning:
            params["alpha_optim"] = self.alpha_optim.state_dict()

        Learner._save_params(self, params, n_episode)

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        Learner.load_params(self, path)

        params = torch.load(path)
        self.actor.load_state_dict(params["actor"])
        self.qf_1.load_state_dict(params["qf_1"])
        self.qf_2.load_state_dict(params["qf_2"])
        self.vf.load_state_dict(params["vf"])
        self.vf_target.load_state_dict(params["vf_target"])
        self.actor_optim.load_state_dict(params["actor_optim"])
        self.qf_1_optim.load_state_dict(params["qf_1_optim"])
        self.qf_2_optim.load_state_dict(params["qf_2_optim"])
        self.vf_optim.load_state_dict(params["vf_optim"])

        if self.hyper_params.auto_entropy_tuning:
            self.alpha_optim.load_state_dict(params["alpha_optim"])

        print("[INFO] loaded the model and optimizer from", path)

    def get_state_dict(self) -> Tuple[OrderedDict]:
        """Return state dicts, mainly for distributed worker."""
        return (self.qf_1.state_dict(), self.qf_2.state_dict(), self.actor.state_dict())

    def get_policy(self) -> nn.Module:
        """Return model (policy) used for action selection."""
        return self.actor
