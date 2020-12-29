import argparse
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
from rl_algorithms.common.networks.brain import Brain, SoftmaxBrain
from rl_algorithms.registry import LEARNERS
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class ACERLearner(Learner):
    def __init__(
        self,
        args: argparse.Namespace,
        env_info: ConfigDict,
        hyper_params,
        log_cfg,
        backbone,
        head,
        optim_cfg,
    ):
        Learner.__init__(self, args, env_info, hyper_params, log_cfg)

        self.backbone_cfg = backbone
        self.head_cfg = head
        self.head_cfg.actor.configs.state_size = (
            self.head_cfg.critic.configs.state_size
        ) = self.env_info.observation_space.shape
        self.head_cfg.actor.configs.output_size = (
            self.head_cfg.critic.configs.output_size
        ) = self.env_info.action_space.n
        self.optim_cfg = optim_cfg
        self._init_network()

    def _init_network(self):
        """Initialize networks and optimizers."""
        self.actor = SoftmaxBrain(self.backbone_cfg.actor, self.head_cfg.actor).to(
            self.device
        )
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

        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def update_model(self, experience: TensorTuple):

        state, action, reward, prob, done, is_first = experience
        state = state.to(self.device)
        action = action.to(self.device)
        prob = prob.to(self.device)
        q = self.critic(state)

        pi = self.actor(state, 1) + 1e-7
        q_a = q.gather(1, action)
        pi_a = pi.gather(1, action)
        v = (q * pi).sum(1).unsqueeze(1).detach()
        rho = pi.detach() / prob
        rho_a = rho.gather(1, action)
        rho_bar = rho_a.clamp(max=self.hyper_params.c)
        correction_coeff = (1 - self.hyper_params.c / rho).clamp(min=0)
        q_ret = self.q_retrace(
            reward, done, q_a, v, rho_bar, is_first, self.hyper_params.gamma
        )
        loss_f = -rho_bar * torch.log(pi_a) * (q_ret - v)
        loss_bc = -correction_coeff * pi.detach() * torch.log(pi) * (q.detach() - v)

        actor_loss = loss_f + loss_bc.sum(1)
        critic_loss = F.smooth_l1_loss(q_a, q_ret)

        self.actor_optim.zero_grad()
        actor_loss.mean().backward()
        self.actor_optim.step()

        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return actor_loss, critic_loss

    def q_retrace(self, reward, done, q_a, v, rho_bar, is_first, gamma):
        q_ret = v[-1] * done[-1]
        q_ret_lst = []

        for i in reversed(range(len(reward))):
            q_ret = reward[i] + gamma * q_ret
            q_ret_lst.append(q_ret.item())
            q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

            if is_first[i] and i != 0:
                q_ret = (
                    v[i - 1] * done[i - 1]
                )  # When a new sequence begins, q_ret is initialized

        q_ret_lst.reverse()
        q_ret = torch.FloatTensor(q_ret_lst).unsqueeze(1).to(self.device)
        return q_ret

    def save_params(self, n_episode):
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
        }
        Learner._save_params(self, params, n_episode)

    def load_params(self, path):
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
