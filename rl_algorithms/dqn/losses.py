# -*- coding: utf-8 -*-
"""Loss functions for DQN.

This module has DQN loss functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.registry import LOSSES
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@LOSSES.register_module
class IQNLoss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return element-wise IQN loss and Q-values.

        Reference: https://github.com/google/dopamine
        """
        states, actions, rewards, next_states, dones = experiences[:5]
        batch_size = states.shape[0]

        # size of rewards: (n_tau_prime_samples x batch_size) x 1.
        rewards = rewards.repeat(head_cfg.configs.n_tau_prime_samples, 1)

        # size of gamma_with_terminal: (n_tau_prime_samples x batch_size) x 1.
        masks = 1 - dones
        gamma_with_terminal = masks * gamma
        gamma_with_terminal = gamma_with_terminal.repeat(
            head_cfg.configs.n_tau_prime_samples, 1
        )
        with torch.no_grad():
            # Get the indices of the maximium Q-value across the action dimension.
            # Shape of replay_next_qt_argmax: (n_tau_prime_samples x batch_size) x 1.
            next_actions = model(next_states).argmax(dim=1)  # double Q
            next_actions = next_actions[:, None]
            next_actions = next_actions.repeat(head_cfg.configs.n_tau_prime_samples, 1)

            # Shape of next_target_values: (n_tau_prime_samples x batch_size) x 1.
            target_quantile_values, _ = target_model.forward_(
                next_states, head_cfg.configs.n_tau_prime_samples
            )
            target_quantile_values = target_quantile_values.gather(1, next_actions)
            target_quantile_values = (
                rewards + gamma_with_terminal * target_quantile_values
            )
            target_quantile_values = target_quantile_values.detach()

            # Reshape to n_tau_prime_samples x batch_size x 1 since this is
            # the manner in which the target_quantile_values are tiled.
            target_quantile_values = target_quantile_values.view(
                head_cfg.configs.n_tau_prime_samples, batch_size, 1
            )

            # Transpose dimensions so that the dimensionality is batch_size x
            # n_tau_prime_samples x 1 to prepare for computation of Bellman errors.
            target_quantile_values = torch.transpose(target_quantile_values, 0, 1)

        # Get quantile values: (n_tau_samples x batch_size) x action_dim.
        quantile_values, quantiles = model.forward_(
            states, head_cfg.configs.n_tau_samples
        )

        reshaped_actions = actions[:, None].repeat(head_cfg.configs.n_tau_samples, 1)
        chosen_action_quantile_values = quantile_values.gather(
            1, reshaped_actions.long()
        )
        chosen_action_quantile_values = chosen_action_quantile_values.view(
            head_cfg.configs.n_tau_samples, batch_size, 1
        )

        # Transpose dimensions so that the dimensionality is batch_size x
        # n_tau_prime_samples x 1 to prepare for computation of Bellman errors.
        chosen_action_quantile_values = torch.transpose(
            chosen_action_quantile_values, 0, 1
        )

        # Shape of bellman_erors and huber_loss:
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        bellman_errors = (
            target_quantile_values[:, :, None, :]
            - chosen_action_quantile_values[:, None, :, :]
        )

        # The huber loss (introduced in QR-DQN) is defined via two cases:
        # case_one: |bellman_errors| <= kappa
        # case_two: |bellman_errors| > kappa
        huber_loss_case_one = (
            (torch.abs(bellman_errors) <= head_cfg.configs.kappa).float()
            * 0.5
            * bellman_errors ** 2
        )
        huber_loss_case_two = (
            (torch.abs(bellman_errors) > head_cfg.configs.kappa).float()
            * head_cfg.configs.kappa
            * (torch.abs(bellman_errors) - 0.5 * head_cfg.configs.kappa)
        )
        huber_loss = huber_loss_case_one + huber_loss_case_two

        # Reshape quantiles to batch_size x num_tau_samples x 1
        quantiles = quantiles.view(head_cfg.configs.n_tau_samples, batch_size, 1)
        quantiles = torch.transpose(quantiles, 0, 1)

        # Tile by num_tau_prime_samples along a new dimension. Shape is now
        # batch_size x num_tau_prime_samples x num_tau_samples x 1.
        # These quantiles will be used for computation of the quantile huber loss
        # below (see section 2.3 of the paper).
        quantiles = quantiles[:, None, :, :].repeat(
            1, head_cfg.configs.n_tau_prime_samples, 1, 1
        )

        # Shape: batch_size x n_tau_prime_samples x n_tau_samples x 1.
        quantile_huber_loss = (
            torch.abs(quantiles - (bellman_errors < 0).float().detach())
            * huber_loss
            / head_cfg.configs.kappa
        )

        # Sum over current quantile value (n_tau_samples) dimension,
        # average over target quantile value (n_tau_prime_samples) dimension.
        # Shape: batch_size x n_tau_prime_samples x 1.
        loss = torch.sum(quantile_huber_loss, dim=2)

        # Shape: batch_size x 1.
        iqn_loss_element_wise = torch.mean(loss, dim=1)

        # q values for regularization.
        q_values = model(states)

        return iqn_loss_element_wise, q_values


@LOSSES.register_module
class C51Loss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return element-wise C51 loss and Q-values."""
        states, actions, rewards, next_states, dones = experiences[:5]
        batch_size = states.shape[0]

        support = torch.linspace(
            head_cfg.configs.v_min, head_cfg.configs.v_max, head_cfg.configs.atom_size
        ).to(device)
        delta_z = float(head_cfg.configs.v_max - head_cfg.configs.v_min) / (
            head_cfg.configs.atom_size - 1
        )

        with torch.no_grad():
            # According to noisynet paper,
            # it resamples noisynet parameters on online network when using double q
            # but we don't because there is no remarkable difference in performance.
            next_actions = model.forward_(next_states)[1].argmax(1)

            next_dist = target_model.forward_(next_states)[0]
            next_dist = next_dist[range(batch_size), next_actions]

            t_z = rewards + (1 - dones) * gamma * support
            t_z = t_z.clamp(min=head_cfg.configs.v_min, max=head_cfg.configs.v_max)
            b = (t_z - head_cfg.configs.v_min) / delta_z
            l = b.floor().long()  # noqa: E741
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (batch_size - 1) * head_cfg.configs.atom_size, batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(batch_size, head_cfg.configs.atom_size)
                .to(device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist, q_values = model.forward_(states)
        log_p = torch.log(
            torch.clamp(dist[range(batch_size), actions.long()], min=1e-7)
        )

        dq_loss_element_wise = -(proj_dist * log_p).sum(1, keepdim=True)

        return dq_loss_element_wise, q_values


@LOSSES.register_module
class DQNLoss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return element-wise dqn loss and Q-values."""
        states, actions, rewards, next_states, dones = experiences[:5]

        q_values = model(states)
        # According to noisynet paper,
        # it resamples noisynet parameters on online network when using double q
        # but we don't because there is no remarkable difference in performance.
        next_q_values = model(next_states)

        next_target_q_values = target_model(next_states)

        curr_q_value = q_values.gather(1, actions.long().unsqueeze(1))
        next_q_value = next_target_q_values.gather(  # Double DQN
            1, next_q_values.argmax(1).unsqueeze(1)
        )

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        target = rewards + gamma * next_q_value * masks

        # calculate dq loss
        dq_loss_element_wise = F.smooth_l1_loss(
            curr_q_value, target.detach(), reduction="none"
        )

        return dq_loss_element_wise, q_values
