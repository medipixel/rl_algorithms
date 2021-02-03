from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.common.noise import GaussianNoise
from rl_algorithms.registry import LEARNERS
from rl_algorithms.utils.config import ConfigDict


@LEARNERS.register_module
class TD3Learner(Learner):
    """Learner for DDPG Agent.

    Attributes:
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        noise_cfg (ConfigDict): config of noise
        target_policy_noise (GaussianNoise): random noise for target values
        actor (nn.Module): actor model to select actions
        critic1 (nn.Module): critic model to predict state values
        critic2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        actor_target (nn.Module): target actor model to select actions
        critic_optim (Optimizer): optimizer for training critic
        actor_optim (Optimizer): optimizer for training actor

    """

    def __init__(
        self,
        hyper_params: ConfigDict,
        log_cfg: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
        optim_cfg: ConfigDict,
        noise_cfg: ConfigDict,
        env_name: str,
        state_size: tuple,
        output_size: int,
        is_test: bool,
        load_from: str,
    ):
        Learner.__init__(self, hyper_params, log_cfg, env_name, is_test)

        self.backbone_cfg = backbone
        self.head_cfg = head
        self.head_cfg.critic.configs.state_size = (state_size[0] + output_size,)
        self.head_cfg.actor.configs.state_size = state_size
        self.head_cfg.actor.configs.output_size = output_size
        self.optim_cfg = optim_cfg
        self.noise_cfg = noise_cfg
        self.load_from = load_from

        self.target_policy_noise = GaussianNoise(
            self.head_cfg.actor.configs.output_size,
            self.noise_cfg.target_policy_noise,
            self.noise_cfg.target_policy_noise,
        )

        self.update_step = 0

        self._init_network()

    def _init_network(self):
        """Initialize networks and optimizers."""
        # create actor
        self.actor = Brain(self.backbone_cfg.actor, self.head_cfg.actor).to(self.device)
        self.actor_target = Brain(self.backbone_cfg.actor, self.head_cfg.actor).to(
            self.device
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        # create critic
        self.critic1 = Brain(self.backbone_cfg.critic, self.head_cfg.critic).to(
            self.device
        )
        self.critic2 = Brain(self.backbone_cfg.critic, self.head_cfg.critic).to(
            self.device
        )

        self.critic_target1 = Brain(self.backbone_cfg.critic, self.head_cfg.critic).to(
            self.device
        )
        self.critic_target2 = Brain(self.backbone_cfg.critic, self.head_cfg.critic).to(
            self.device
        )

        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # concat critic parameters to use one optim
        critic_parameters = list(self.critic1.parameters()) + list(
            self.critic2.parameters()
        )

        # create optimizers
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.critic_optim = optim.Adam(
            critic_parameters,
            lr=self.optim_cfg.lr_critic,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # load the optimizer and model parameters
        if self.load_from is not None:
            self.load_params(self.load_from)

    def update_model(
        self, experience: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, ...]:
        """Update TD3 actor and critic networks."""
        self.update_step += 1

        states, actions, rewards, next_states, dones = experience
        masks = 1 - dones

        # get actions with noise
        noise = common_utils.numpy2floattensor(
            self.target_policy_noise.sample(), self.device
        )
        clipped_noise = torch.clamp(
            noise,
            -self.noise_cfg.target_policy_noise_clip,
            self.noise_cfg.target_policy_noise_clip,
        )
        next_actions = (self.actor_target(next_states) + clipped_noise).clamp(-1.0, 1.0)

        # min (Q_1', Q_2')
        next_states_actions = torch.cat((next_states, next_actions), dim=-1)
        next_values1 = self.critic_target1(next_states_actions)
        next_values2 = self.critic_target2(next_states_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + self.hyper_params.gamma * next_values * masks
        curr_returns = curr_returns.detach()

        # critic loss
        state_actions = torch.cat((states, actions), dim=-1)
        values1 = self.critic1(state_actions)
        values2 = self.critic2(state_actions)
        critic1_loss = F.mse_loss(values1, curr_returns)
        critic2_loss = F.mse_loss(values2, curr_returns)

        # train critic
        critic_loss = critic1_loss + critic2_loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.update_step % self.hyper_params.policy_update_freq == 0:
            # policy loss
            actions = self.actor(states)
            state_actions = torch.cat((states, actions), dim=-1)
            actor_loss = -self.critic1(state_actions).mean()

            # train actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            tau = self.hyper_params.tau
            common_utils.soft_update(self.critic1, self.critic_target1, tau)
            common_utils.soft_update(self.critic2, self.critic_target2, tau)
            common_utils.soft_update(self.actor, self.actor_target, tau)
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic_target1": self.critic_target1.state_dict(),
            "critic_target2": self.critic_target2.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }

        Learner._save_params(self, params, n_episode)

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        Learner.load_params(self, path)

        params = torch.load(path)
        self.critic1.load_state_dict(params["critic1"])
        self.critic2.load_state_dict(params["critic2"])
        self.critic_target1.load_state_dict(params["critic_target1"])
        self.critic_target2.load_state_dict(params["critic_target2"])
        self.critic_optim.load_state_dict(params["critic_optim"])
        self.actor.load_state_dict(params["actor"])
        self.actor_target.load_state_dict(params["actor_target"])
        self.actor_optim.load_state_dict(params["actor_optim"])
        print("[INFO] loaded the model and optimizer from", path)

    def get_state_dict(self) -> Tuple[OrderedDict]:
        """Return state dicts, mainly for distributed worker."""
        return (
            self.critic_target1.state_dict(),
            self.critic_target2.state_dict(),
            self.actor.state_dict(),
        )

    def get_policy(self) -> nn.Module:
        """Return model (policy) used for action selection."""
        return self.actor
