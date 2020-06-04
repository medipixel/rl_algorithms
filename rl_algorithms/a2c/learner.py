import argparse
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class A2CLearner(Learner):
    """Learner for A2C Agent

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
        """Update A2C actor and critic networks"""
        actor, critic = networks
        actor_optim, critic_optim = optimizer

        log_prob, pred_value, next_state, reward, done = experience
        next_state = torch.FloatTensor(next_state).to(self.device)

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        next_value = critic(next_state).detach()
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

        actor_optim.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(actor.parameters(), gradient_clip_ac)
        actor_optim.step()

        critic_optim.zero_grad()
        value_loss.backward()
        clip_grad_norm_(critic.parameters(), gradient_clip_cr)
        critic_optim.step()

        return policy_loss.item(), value_loss.item()
