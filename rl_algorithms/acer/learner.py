from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.registry import LEARNERS
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class ACERLearner(Learner):
    """Learner for ACER Agent.

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        model (nn.Module): model to select actions and predict values
        model_optim (Optimizer): optimizer for training model

    """

    def __init__(
        self,
        backbone: ConfigDict,
        head: ConfigDict,
        optim_cfg: ConfigDict,
        trust_region: ConfigDict,
        hyper_params: ConfigDict,
        log_cfg: ConfigDict,
        env_info: ConfigDict,
        is_test: bool,
        load_from: str,
    ):
        Learner.__init__(self, hyper_params, log_cfg, env_info.name, is_test)

        self.backbone_cfg = backbone
        self.head_cfg = head
        self.load_from = load_from
        self.head_cfg.actor.configs.state_size = env_info.observation_space.shape
        self.head_cfg.critic.configs.state_size = env_info.observation_space.shape
        self.head_cfg.actor.configs.output_size = env_info.action_space.n
        self.head_cfg.critic.configs.output_size = env_info.action_space.n
        self.optim_cfg = optim_cfg
        self.gradient_clip = hyper_params.gradient_clip
        self.trust_region = trust_region

        self._init_network()

    def _init_network(self):
        """Initialize network and optimizer."""
        self.actor = Brain(self.backbone_cfg.actor, self.head_cfg.actor).to(self.device)
        self.critic = Brain(self.backbone_cfg.critic, self.head_cfg.critic).to(
            self.device
        )
        # create optimizer
        self.actor_optim = optim.Adam(
            self.actor.parameters(), lr=self.optim_cfg.lr, eps=self.optim_cfg.adam_eps
        )
        self.critic_optim = optim.Adam(
            self.critic.parameters(), lr=self.optim_cfg.lr, eps=self.optim_cfg.adam_eps
        )

        self.actor_target = Brain(self.backbone_cfg.actor, self.head_cfg.actor).to(
            self.device
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        if self.load_from is not None:
            self.load_params(self.load_from)

    def update_model(self, experience: Tuple) -> torch.Tensor:

        state, action, reward, prob, done = experience
        state = state.to(self.device)
        reward = reward.to(self.device)
        action = action.to(self.device)
        prob = prob.to(self.device).squeeze()
        done = done.to(self.device)

        pi = F.softmax(self.actor(state), 1)

        q = self.critic(state)
        q_i = q.gather(1, action)
        pi_i = pi.gather(1, action)

        with torch.no_grad():
            v = (q * pi).sum(1).unsqueeze(1)
            rho = pi / (prob + 1e-8)
        rho_i = rho.gather(1, action)
        rho_bar = rho_i.clamp(max=self.hyper_params.c)

        q_ret = self.q_retrace(
            reward, done, q_i, v, rho_bar, self.hyper_params.gamma
        ).to(self.device)

        loss_f = -rho_bar * torch.log(pi_i + 1e-8) * (q_ret - v)
        loss_bc = (
            -(1 - (self.hyper_params.c / rho)).clamp(min=0)
            * pi.detach()
            * torch.log(pi + 1e-8)
            * (q.detach() - v)
        )

        value_loss = torch.sqrt((q_i - q_ret).pow(2)).mean() * 0.5

        if self.trust_region.use_trust_region:
            g = loss_f + loss_bc
            pi_target = F.softmax(self.actor_target(state), 1)
            # gradient of partial Q KL(P || Q) = - P / Q
            k = -pi_target / (pi + 1e-8)
            k_dot_g = k * g
            tr = (
                g
                - ((k_dot_g - self.trust_region.delta) / torch.norm(k)).clamp(max=0) * k
            )
            loss = tr.mean() + value_loss
        else:
            loss = loss_f.mean() + loss_bc.sum(1).mean() + value_loss

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        for name, param in self.actor.named_parameters():
            if not torch.isfinite(param.grad).all():
                print(name, torch.isfinite(param.grad).all())
                print("Warning : Gradient is infinite. Do not update gradient.")
                return loss
        for name, param in self.critic.named_parameters():
            if not torch.isfinite(param.grad).all():
                print(name, torch.isfinite(param.grad).all())
                print("Warning : Gradient is infinite. Do not update gradient.")
                return loss
        self.actor_optim.step()
        self.critic_optim.step()

        common_utils.soft_update(self.actor, self.actor_target, self.hyper_params.tau)

        return loss

    @staticmethod
    def q_retrace(
        reward: torch.Tensor,
        done: torch.Tensor,
        q_a: torch.Tensor,
        v: torch.Tensor,
        rho_bar: torch.Tensor,
        gamma: float,
    ):
        """Calculate Q retrace."""
        q_ret = v[-1]
        q_ret_lst = []

        for i in reversed(range(len(reward))):
            q_ret = reward[i] + gamma * q_ret * done[i]
            q_ret_lst.append(q_ret.item())
            q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]

        q_ret_lst.reverse()
        q_ret = torch.FloatTensor(q_ret_lst).unsqueeze(1)
        return q_ret

    def save_params(self, n_episode: int):
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
        }
        Learner._save_params(self, params, n_episode)

    def load_params(self, path: str):
        Learner.load_params(self, path)

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] Loaded the model and optimizer from", path)

    def get_state_dict(self) -> Tuple[OrderedDict]:
        """Return state dicts, mainly for distributed worker."""
        return (self.model.state_dict(), self.optim.state_dict())

    def get_policy(self) -> nn.Module:
        """Return model (policy) used for action selection."""
        return self.actor
