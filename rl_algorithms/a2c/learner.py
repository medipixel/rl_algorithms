from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.registry import LEARNERS
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class A2CLearner(Learner):
    """Learner for A2C Agent.

    Attributes:
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        actor (nn.Module): actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optim (Optimizer): optimizer for training actor
        critic_optim (Optimizer): optimizer for training critic

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
            self.head_cfg.critic.configs.state_size
        ) = state_size
        self.head_cfg.actor.configs.output_size = output_size
        self.optim_cfg = optim_cfg
        self.load_from = load_from

        self._init_network()

    def _init_network(self):
        """Initialize networks and optimizers."""
        self.actor = Brain(self.backbone_cfg.actor, self.head_cfg.actor).to(self.device)
        self.critic = Brain(self.backbone_cfg.critic, self.head_cfg.critic).to(
            self.device
        )

        # create optimizer
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=self.optim_cfg.lr_critic,
            weight_decay=self.optim_cfg.weight_decay,
        )

        if self.load_from is not None:
            self.load_params(self.load_from)

    def update_model(self, experience: TensorTuple) -> TensorTuple:
        """Update A2C actor and critic networks"""

        log_prob, pred_value, next_state, reward, done = experience
        next_state = numpy2floattensor(next_state, self.device)

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        next_value = self.critic(next_state).detach()
        q_value = reward + self.hyper_params.gamma * next_value * mask
        q_value = q_value.to(self.device)

        # advantage = Q_t - V(s_t)
        advantage = q_value - pred_value

        # calculate loss at the current step
        policy_loss = -advantage.detach() * log_prob  # adv. is not backpropagated
        policy_loss += self.hyper_params.w_entropy * -log_prob  # entropy
        value_loss = F.smooth_l1_loss(pred_value, q_value.detach())

        # train
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        self.actor_optim.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        value_loss.backward()
        clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        return policy_loss.item(), value_loss.item()

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
        }

        Learner._save_params(self, params, n_episode)

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        Learner.load_params(self, path)

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] Loaded the model and optimizer from", path)

    def get_state_dict(self) -> Tuple[OrderedDict]:
        """Return state dicts, mainly for distributed worker."""
        return (self.critic.state_dict(), self.actor.state_dict())

    def get_policy(self) -> nn.Module:
        """Return model (policy) used for action selection."""
        return self.actor
