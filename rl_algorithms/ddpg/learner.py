import argparse
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class DDPGLearner(Learner):
    """Learner for DQN Agent

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters

    """

    def __init__(
        self, args: argparse.Namespace, hyper_params: ConfigDict, device: torch.device
    ):
        Learner.__init__(self, args, hyper_params, device)

    def update_model(
        self,
        networks: Tuple[Brain, ...],
        optimizer: Union[optim.Optimizer, Tuple[optim.Optimizer, ...]],
        experience: TensorTuple,
    ) -> TensorTuple:
        """Update ddpg actor and critic networks"""
        actor, actor_target, critic, critic_target = networks
        actor_optim, critic_optim = optimizer
        states, actions, rewards, next_states, dones = experience

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = actor_target(next_states)
        next_values = critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + self.hyper_params.gamma * next_values * masks
        curr_returns = curr_returns.to(self.device)

        # train critic
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        values = critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)
        critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(critic.parameters(), gradient_clip_cr)
        critic_optim.step()

        # train actor
        actions = actor(states)
        actor_loss = -critic(torch.cat((states, actions), dim=-1)).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(actor.parameters(), gradient_clip_ac)
        actor_optim.step()

        # update target networks
        common_utils.soft_update(actor, actor_target, self.hyper_params.tau)
        common_utils.soft_update(critic, critic_target, self.hyper_params.tau)

        return actor_loss.item(), critic_loss.item()
