import argparse
from typing import Tuple, Union

import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
from rl_algorithms.common.networks.brain import Brain
import rl_algorithms.ppo.utils as ppo_utils
from rl_algorithms.utils.config import ConfigDict


class PPOLearner(Learner):
    """Learner for PPO Agent

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters

    """

    def __init__(
        self,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        device: torch.device,
        is_discrete: bool,
    ):
        Learner.__init__(self, args, hyper_params, device)

        self.is_discrete = is_discrete

    def update_model(
        self,
        networks: Tuple[Brain, ...],
        optimizer: Union[optim.Optimizer, Tuple[optim.Optimizer, ...]],
        experience: TensorTuple,
        epsilon: float,
    ) -> TensorTuple:
        """Update PPO actor and critic networks"""
        actor, critic = networks
        actor_optim, critic_optim = optimizer

        states, actions, rewards, values, log_probs, next_state, masks = experience
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = critic(next_state)

        returns = ppo_utils.compute_gae(
            next_value,
            rewards,
            masks,
            values,
            self.hyper_params.gamma,
            self.hyper_params.tau,
        )

        states = torch.cat(states)
        actions = torch.cat(actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(values).detach()
        log_probs = torch.cat(log_probs).detach()
        advantages = returns - values

        if self.is_discrete:
            actions = actions.unsqueeze(1)
            log_probs = log_probs.unsqueeze(1)

        if self.hyper_params.standardize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        actor_losses, critic_losses, total_losses = [], [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_utils.ppo_iter(
            self.hyper_params.epoch,
            self.hyper_params.batch_size,
            states,
            actions,
            values,
            log_probs,
            returns,
            advantages,
        ):
            # calculate ratios
            _, dist = actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            surr_loss = ratio * adv
            clipped_surr_loss = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
            actor_loss = -torch.min(surr_loss, clipped_surr_loss).mean()

            # critic_loss
            value = critic(state)
            if self.hyper_params.use_clipped_value_loss:
                value_pred_clipped = old_value + torch.clamp(
                    (value - old_value), -epsilon, epsilon
                )
                value_loss_clipped = (return_ - value_pred_clipped).pow(2)
                value_loss = (return_ - value).pow(2)
                critic_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()
            else:
                critic_loss = 0.5 * (return_ - value).pow(2).mean()

            # entropy
            entropy = dist.entropy().mean()

            # total_loss
            w_value = self.hyper_params.w_value
            w_entropy = self.hyper_params.w_entropy

            total_loss = actor_loss + w_value * critic_loss - w_entropy * entropy

            # train critic
            gradient_clip_ac = self.hyper_params.gradient_clip_ac
            gradient_clip_cr = self.hyper_params.gradient_clip_cr

            critic_optim.zero_grad()
            total_loss.backward(retain_graph=True)
            clip_grad_norm_(critic.parameters(), gradient_clip_ac)
            critic_optim.step()

            # train actor
            actor_optim.zero_grad()
            total_loss.backward()
            clip_grad_norm_(critic.parameters(), gradient_clip_cr)
            actor_optim.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            total_losses.append(total_loss.item())

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        total_loss = sum(total_losses) / len(total_losses)

        return actor_loss, critic_loss, total_loss
