import argparse
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner
from rl_algorithms.common.networks.brain import ACERBrain
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
        args: argparse.Namespace,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        log_cfg: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
        optim_cfg: ConfigDict,
    ):
        Learner.__init__(self, args, env_info, hyper_params, log_cfg)

        self.backbone_cfg = backbone
        self.head_cfg = head
        self.head_cfg.configs.state_size = self.env_info.observation_space.shape
        self.head_cfg.configs.output_size = self.env_info.action_space.n
        self.optim_cfg = optim_cfg
        self.use_n_step = False
        self._init_network()

    def _init_network(self):
        """Initialize network and optimizer."""
        self.model = ACERBrain(self.backbone_cfg, self.head_cfg).to(self.device)

        # create optimizer
        self.optim = optim.RMSprop(self.model.parameters(), lr=self.optim_cfg.lr,)

        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def update_model(self, experience: Tuple) -> torch.Tensor:

        state, action, reward, prob, done = experience
        state = state.to(self.device).squeeze()
        action = action.to(self.device).type(torch.int64).transpose(0, 1)
        prob = prob.to(self.device).squeeze()
        reward = reward.squeeze()
        done = done.squeeze()

        q = self.model.q(state)
        q_i = q.gather(1, action)

        pi = self.model.pi(state, 1)
        pi_i = pi.gather(1, action)

        with torch.no_grad():
            v = (q * pi).sum(1)
            rho = pi / (prob + 1e-8)

        rho_i = rho.gather(0, action)
        rho_bar = rho_i.clamp(max=self.hyper_params.c)

        q_ret = self.q_retrace(
            reward, done, q_i, v, rho_bar, self.hyper_params.gamma
        ).to(self.device)

        loss_f = -rho_bar * torch.log(pi_i + 1e-8) * (q_ret - v)
        loss_bc = (
            -(1 - (self.hyper_params.c / rho)).clamp(min=0)
            * pi.detach()
            * torch.log(pi + 1e-8)
            * (q.detach() - v.unsqueeze(-1))
        )

        value_loss = torch.sqrt((q_i - q_ret).pow(2)).mean() * 0.5

        loss = loss_f.mean() + loss_bc.sum(1).mean() + value_loss

        self.optim.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), 10)

        for name, param in self.model.named_parameters():
            if not torch.isfinite(param.grad).all():
                print(name, torch.isfinite(param.grad).all())
                print("Warning : Gradient is infinite. Do not update gradient.")
                return loss
        self.optim.step()

        return loss

    @staticmethod
    def q_retrace(
        reward: list,
        done: list,
        q_a: torch.Tensor,
        v: torch.Tensor,
        rho_bar: torch.Tensor,
        gamma: float,
    ):
        """Calculate Q retrace."""
        q_ret = v[-1] * done[-1]
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
            "model_state_dict": self.model.state_dict(),
            "model_optim_state_dict": self.optim.state_dict(),
        }
        Learner._save_params(self, params, n_episode)

    def load_params(self, path: str):
        Learner.load_params(self, path)

        params = torch.load(path)
        self.model.load_state_dict(params["model_state_dict"])
        self.optim.load_state_dict(params["model_optim_state_dict"])
        print("[INFO] Loaded the model and optimizer from", path)

    def get_state_dict(self) -> Tuple[OrderedDict]:
        """Return state dicts, mainly for distributed worker."""
        return (self.model.state_dict(), self.optim.state_dict())

    def get_policy(self) -> nn.Module:
        """Return model (policy) used for action selection."""
        return self.model.pi
