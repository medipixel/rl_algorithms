import argparse
from typing import Tuple, Union

import torch
import torch.nn.functional as F
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.common.noise import GaussianNoise
from rl_algorithms.utils.config import ConfigDict


class TD3Learner(Learner):
    """Learner for DDPG Agent

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters

    """

    def __init__(
        self,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        device: torch.device,
        noise_cfg: ConfigDict,
        target_policy_noise: GaussianNoise,
    ):
        Learner.__init__(self, args, hyper_params, device)

        self.update_step = 0
        self.noise_cfg = noise_cfg
        self.target_policy_noise = target_policy_noise

    def update_model(
        self,
        networks: Tuple[Brain, ...],
        optimizer: Union[optim.Optimizer, Tuple[optim.Optimizer, ...]],
        experience: TensorTuple,
    ) -> TensorTuple:
        """Update TD3 actor and critic networks"""
        actor, actor_target, critic1, critic_target1, critic2, critic_target2 = networks
        actor_optim, critic_optim = optimizer
        states, actions, rewards, next_states, dones = experience
        masks = 1 - dones

        # get actions with noise
        noise = torch.FloatTensor(self.target_policy_noise.sample()).to(self.device)
        clipped_noise = torch.clamp(
            noise,
            -self.noise_cfg.target_policy_noise_clip,
            self.noise_cfg.target_policy_noise_clip,
        )
        next_actions = (actor_target(next_states) + clipped_noise).clamp(-1.0, 1.0)

        # min (Q_1', Q_2')
        next_states_actions = torch.cat((next_states, next_actions), dim=-1)
        next_values1 = critic_target1(next_states_actions)
        next_values2 = critic_target2(next_states_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + self.hyper_params.gamma * next_values * masks
        curr_returns = curr_returns.detach()

        # critic loss
        state_actions = torch.cat((states, actions), dim=-1)
        values1 = critic1(state_actions)
        values2 = critic2(state_actions)
        critic1_loss = F.mse_loss(values1, curr_returns)
        critic2_loss = F.mse_loss(values2, curr_returns)

        # train critic
        critic_loss = critic1_loss + critic2_loss
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        if self.update_step % self.hyper_params.policy_update_freq == 0:
            # policy loss
            actions = actor(states)
            state_actions = torch.cat((states, actions), dim=-1)
            actor_loss = -critic1(state_actions).mean()

            # train actor
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # update target networks
            tau = self.hyper_params.tau
            common_utils.soft_update(critic1, critic_target1, tau)
            common_utils.soft_update(critic2, critic_target2, tau)
            common_utils.soft_update(actor, actor_target, tau)
        else:
            actor_loss = torch.zeros(1)

        self.update_step = self.update_step + 1

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()
