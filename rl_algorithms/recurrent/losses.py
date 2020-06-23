# -*- coding: utf-8 -*-
"""Loss functions for R2D1.

This module has R2D1 loss functions.

- Author: Euijin Jeong
- Contact: euijin.jeong@medipixel.io
"""

from typing import Tuple

import torch
from torch.nn import functional as F

from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.recurrent.utils import slice_r2d1_arguments, valid_from_done
from rl_algorithms.registry import LOSSES
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@LOSSES.register_module
class R2D1DQNLoss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict,
        burn_in_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return R2D1DQN loss and Q-values."""
        # TODO: Combine with DQNLoss
        output_size = head_cfg.configs.output_size
        (
            burnin_states_tuple,
            states_tuple,
            burnin_prev_actions_tuple,
            agent_actions,
            prev_actions_tuple,
            burnin_prev_rewards_tuple,
            agent_rewards,
            prev_rewards_tuple,
            burnin_dones_tuple,
            agent_dones,
            init_rnn_state,
        ) = slice_r2d1_arguments(experiences, burn_in_step, output_size)

        with torch.no_grad():
            _, target_rnn_state = target_model(
                burnin_states_tuple[1],
                init_rnn_state,
                burnin_prev_actions_tuple[1],
                burnin_prev_rewards_tuple[1],
            )
            _, init_rnn_state = model(
                burnin_states_tuple[0],
                init_rnn_state,
                burnin_prev_actions_tuple[0],
                burnin_prev_rewards_tuple[0],
            )

            init_rnn_state = torch.transpose(init_rnn_state, 0, 1)
            target_rnn_state = torch.transpose(target_rnn_state, 0, 1)

        burnin_invalid_mask = valid_from_done(burnin_dones_tuple[0].transpose(0, 1))
        burnin_target_invalid_mask = valid_from_done(
            burnin_dones_tuple[1].transpose(0, 1)
        )
        init_rnn_state[burnin_invalid_mask] = 0
        target_rnn_state[burnin_target_invalid_mask] = 0

        q_values, _ = model(
            states_tuple[0],
            init_rnn_state,
            prev_actions_tuple[0],
            prev_rewards_tuple[0],
        )
        q_value = q_values.gather(-1, agent_actions)

        with torch.no_grad():
            target_q_values, _ = target_model(
                states_tuple[1],
                target_rnn_state,
                prev_actions_tuple[1],
                prev_rewards_tuple[1],
            )
            next_q_values, _ = model(
                states_tuple[1],
                target_rnn_state,
                prev_actions_tuple[0],
                prev_rewards_tuple[0],
            )
            next_action = torch.argmax(next_q_values, dim=-1)
            target_q_value = target_q_values.gather(-1, next_action.unsqueeze(-1))

        target = agent_rewards + gamma * target_q_value * (1 - agent_dones)
        dq_loss_element_wise = F.smooth_l1_loss(
            q_value, target.detach(), reduction="none"
        )
        delta = abs(torch.mean(dq_loss_element_wise, dim=1))

        return delta, q_value


@LOSSES.register_module
class R2D1C51Loss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict,
        burn_in_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return element-wise C51 loss and Q-values."""
        # TODO: Combine with IQNLoss
        output_size = head_cfg.configs.output_size
        (
            burnin_states_tuple,
            states_tuple,
            burnin_prev_actions_tuple,
            agent_actions,
            prev_actions_tuple,
            burnin_prev_rewards_tuple,
            agent_rewards,
            prev_rewards_tuple,
            burnin_dones_tuple,
            agent_dones,
            init_rnn_state,
        ) = slice_r2d1_arguments(experiences, burn_in_step, output_size)

        batch_size = states_tuple[0].shape[0]
        sequence_size = states_tuple[0].shape[1]

        with torch.no_grad():
            _, target_rnn_state = target_model(
                burnin_states_tuple[1],
                init_rnn_state,
                burnin_prev_actions_tuple[1],
                burnin_prev_rewards_tuple[1],
            )
            _, init_rnn_state = model(
                burnin_states_tuple[0],
                init_rnn_state,
                burnin_prev_actions_tuple[0],
                burnin_prev_rewards_tuple[0],
            )

            init_rnn_state = torch.transpose(init_rnn_state, 0, 1)
            target_rnn_state = torch.transpose(target_rnn_state, 0, 1)

        burnin_invalid_mask = valid_from_done(burnin_dones_tuple[0].transpose(0, 1))
        burnin_target_invalid_mask = valid_from_done(
            burnin_dones_tuple[1].transpose(0, 1)
        )
        init_rnn_state[burnin_invalid_mask] = 0
        target_rnn_state[burnin_target_invalid_mask] = 0

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
            next_actions, _ = model.forward_(
                states_tuple[1],
                target_rnn_state,
                prev_actions_tuple[1],
                prev_rewards_tuple[1],
            )
            next_actions = next_actions[1].argmax(-1)
            next_dist, _ = target_model.forward_(
                states_tuple[1],
                target_rnn_state,
                prev_actions_tuple[1],
                prev_rewards_tuple[1],
            )
            next_dist = next_dist[0][range(batch_size * sequence_size), next_actions]

            t_z = agent_rewards + (1 - agent_dones) * gamma * support
            t_z = t_z.clamp(min=head_cfg.configs.v_min, max=head_cfg.configs.v_max)
            b = (t_z - head_cfg.configs.v_min) / delta_z
            b = b.view(batch_size * sequence_size, -1)
            l = b.floor().long()  # noqa: E741
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0,
                    (batch_size * sequence_size - 1) * head_cfg.configs.atom_size,
                    batch_size * sequence_size,
                )
                .long()
                .unsqueeze(1)
            )
            offset = offset.expand(
                batch_size * sequence_size, head_cfg.configs.atom_size
            ).to(device)
            proj_dist = torch.zeros(next_dist.size(), device=device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        (dist, q_values), _ = model.forward_(
            states_tuple[0],
            init_rnn_state,
            prev_actions_tuple[0],
            prev_rewards_tuple[0],
        )
        log_p = dist[
            range(batch_size * sequence_size),
            agent_actions.contiguous().view(batch_size * sequence_size).long(),
        ]
        log_p = torch.log(log_p.clamp(min=1e-5))
        log_p = log_p.view(batch_size, sequence_size, -1)
        proj_dist = proj_dist.view(batch_size, sequence_size, -1)
        dq_loss_element_wise = -(proj_dist * log_p).sum(-1).mean(1)

        return dq_loss_element_wise, q_values


@LOSSES.register_module
class R2D1IQNLoss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict,
        burn_in_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return R2D1 loss and Q-values."""
        # TODO: Combine with IQNLoss
        output_size = head_cfg.configs.output_size
        (
            burnin_states_tuple,
            states_tuple,
            burnin_prev_actions_tuple,
            agent_actions,
            prev_actions_tuple,
            burnin_prev_rewards_tuple,
            agent_rewards,
            prev_rewards_tuple,
            burnin_dones_tuple,
            agent_dones,
            init_rnn_state,
        ) = slice_r2d1_arguments(experiences, burn_in_step, output_size)

        batch_size = states_tuple[0].shape[0]
        sequence_size = states_tuple[0].shape[1]

        with torch.no_grad():
            _, target_rnn_state = target_model(
                burnin_states_tuple[1],
                init_rnn_state,
                burnin_prev_actions_tuple[1],
                burnin_prev_rewards_tuple[1],
            )
            _, init_rnn_state = model(
                burnin_states_tuple[0],
                init_rnn_state,
                burnin_prev_actions_tuple[0],
                burnin_prev_rewards_tuple[0],
            )

            init_rnn_state = torch.transpose(init_rnn_state, 0, 1)
            target_rnn_state = torch.transpose(target_rnn_state, 0, 1)

        burnin_invalid_mask = valid_from_done(burnin_dones_tuple[0].transpose(0, 1))
        burnin_target_invalid_mask = valid_from_done(
            burnin_dones_tuple[1].transpose(0, 1)
        )
        init_rnn_state[burnin_invalid_mask] = 0
        target_rnn_state[burnin_target_invalid_mask] = 0

        # size of rewards: (n_tau_prime_samples x batch_size) x 1.
        agent_rewards = agent_rewards.repeat(head_cfg.configs.n_tau_prime_samples, 1, 1)

        # size of gamma_with_terminal: (n_tau_prime_samples x batch_size) x 1.
        masks = 1 - agent_dones
        gamma_with_terminal = masks * gamma
        gamma_with_terminal = gamma_with_terminal.repeat(
            head_cfg.configs.n_tau_prime_samples, 1, 1
        )
        # Get the indices of the maximium Q-value across the action dimension.
        # Shape of replay_next_qt_argmax: (n_tau_prime_samples x batch_size) x 1.
        next_actions, _ = model(
            states_tuple[1],
            target_rnn_state,
            prev_actions_tuple[1],
            prev_rewards_tuple[1],
        ).argmax(dim=-1)
        next_actions = next_actions[:, :, None]
        next_actions = next_actions.repeat(head_cfg.configs.n_tau_prime_samples, 1, 1)

        with torch.no_grad():
            # Shape of next_target_values: (n_tau_prime_samples x batch_size) x 1.
            target_quantile_values, _, _ = target_model.forward_(
                states_tuple[1],
                target_rnn_state,
                prev_actions_tuple[1],
                prev_rewards_tuple[1],
                head_cfg.configs.n_tau_prime_samples,
            )
            target_quantile_values = target_quantile_values.gather(-1, next_actions)
            target_quantile_values = (
                agent_rewards + gamma_with_terminal * target_quantile_values
            )
            target_quantile_values = target_quantile_values.detach()

            # Reshape to n_tau_prime_samples x batch_size x 1 since this is
            # the manner in which the target_quantile_values are tiled.
            target_quantile_values = target_quantile_values.view(
                head_cfg.configs.n_tau_prime_samples, batch_size, sequence_size, 1
            )

            # Transpose dimensions so that the dimensionality is batch_size x
            # n_tau_prime_samples x 1 to prepare for computation of Bellman errors.
            target_quantile_values = torch.transpose(target_quantile_values, 0, 1)

        # Get quantile values: (n_tau_samples x batch_size) x action_dim.
        quantile_values, quantiles, _ = model.forward_(
            states_tuple[0],
            init_rnn_state,
            prev_actions_tuple[0],
            prev_rewards_tuple[0],
            head_cfg.configs.n_tau_samples,
        )

        reshaped_actions = agent_actions.repeat(head_cfg.configs.n_tau_samples, 1, 1)
        chosen_action_quantile_values = quantile_values.gather(
            -1, reshaped_actions.long()
        )
        chosen_action_quantile_values = chosen_action_quantile_values.view(
            head_cfg.configs.n_tau_samples, batch_size, sequence_size, 1
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
        quantiles = quantiles.view(
            head_cfg.configs.n_tau_samples, batch_size, sequence_size, 1
        )
        quantiles = torch.transpose(quantiles, 0, 1)

        # Tile by num_tau_prime_samples along a new dimension. Shape is now
        # batch_size x num_tau_prime_samples x num_tau_samples x sequence_length x 1.
        # These quantiles will be used for computation of the quantile huber loss
        # below (see section 2.3 of the paper).
        quantiles = quantiles[:, None, :, :, :].repeat(
            1, head_cfg.configs.n_tau_prime_samples, 1, 1, 1
        )

        # Shape: batch_size x n_tau_prime_samples x n_tau_samples x sequence_length x 1.
        quantile_huber_loss = (
            torch.abs(quantiles - (bellman_errors < 0).float().detach())
            * huber_loss
            / head_cfg.configs.kappa
        )

        # Sum over current quantile value (n_tau_samples) dimension,
        # average over target quantile value (n_tau_prime_samples) dimension.
        # Shape: batch_size x n_tau_prime_samples x 1.
        loss = torch.sum(quantile_huber_loss, dim=2)

        # Shape: batch_size x sequence_length x 1.
        iqn_loss_element_wise = torch.mean(loss, dim=1)

        # Shape: batch_size x 1.
        iqn_loss_element_wise = abs(torch.mean(iqn_loss_element_wise, dim=1))

        # q values for regularization.
        q_values, _ = model(
            states_tuple[0],
            init_rnn_state,
            prev_actions_tuple[0],
            prev_rewards_tuple[0],
        )

        return iqn_loss_element_wise, q_values
