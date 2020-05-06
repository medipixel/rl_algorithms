# -*- coding: utf-8 -*-
"""Loss functions for DQN.

This module has DQN loss functions.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from rl_algorithms.common.helper_functions import make_one_hot, valid_from_done
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
        target_quantile_values = rewards + gamma_with_terminal * target_quantile_values
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
        quantiles = quantiles.to(device)

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
        log_p = torch.log(dist[range(batch_size), actions.long()])

        dq_loss_element_wise = -(proj_dist * log_p).sum(1)

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
        target = target.to(device)

        # calculate dq loss
        dq_loss_element_wise = F.smooth_l1_loss(
            curr_q_value, target.detach(), reduction="none"
        )

        return dq_loss_element_wise, q_values


def slice_r2d2_arguments(experiences, head_cfg):
    states, actions, rewards, hiddens, dones = experiences[:5]

    burnin_states = states[:, 1 : head_cfg.configs.burn_in_step]
    target_burnin_states = states[:, 1 : head_cfg.configs.burn_in_step + 1]
    agent_states = states[:, head_cfg.configs.burn_in_step : -1]
    target_states = states[:, head_cfg.configs.burn_in_step + 1 :]

    burnin_prev_actions = make_one_hot(
        actions[:, : head_cfg.configs.burn_in_step - 1], head_cfg.configs.output_size,
    )
    target_burnin_prev_actions = make_one_hot(
        actions[:, : head_cfg.configs.burn_in_step], head_cfg.configs.output_size
    )
    agent_actions = actions[:, head_cfg.configs.burn_in_step : -1].long().unsqueeze(-1)
    prev_actions = make_one_hot(
        actions[:, head_cfg.configs.burn_in_step - 1 : -2],
        head_cfg.configs.output_size,
    )
    target_prev_actions = make_one_hot(
        actions[:, head_cfg.configs.burn_in_step : -1].long(),
        head_cfg.configs.output_size,
    )

    burnin_prev_rewards = rewards[:, : head_cfg.configs.burn_in_step - 1].unsqueeze(-1)
    target_burnin_prev_rewards = rewards[:, : head_cfg.configs.burn_in_step].unsqueeze(
        -1
    )
    agent_rewards = rewards[:, head_cfg.configs.burn_in_step : -1].unsqueeze(-1)
    prev_rewards = rewards[:, head_cfg.configs.burn_in_step - 1 : -2].unsqueeze(-1)
    target_prev_rewards = agent_rewards
    burnin_dones = dones[:, 1 : head_cfg.configs.burn_in_step].unsqueeze(-1)
    burnin_target_dones = dones[:, 1 : head_cfg.configs.burn_in_step + 1].unsqueeze(-1)
    agent_dones = dones[:, head_cfg.configs.burn_in_step : -1].unsqueeze(-1)
    init_rnn_state = hiddens[:, 0].squeeze(1).contiguous()

    return (
        burnin_states,
        target_burnin_states,
        agent_states,
        target_states,
        burnin_prev_actions,
        target_burnin_prev_actions,
        agent_actions,
        prev_actions,
        target_prev_actions,
        burnin_prev_rewards,
        target_burnin_prev_rewards,
        agent_rewards,
        prev_rewards,
        target_prev_rewards,
        burnin_dones,
        burnin_target_dones,
        agent_dones,
        init_rnn_state,
    )


@LOSSES.register_module
class R2D1DQNLoss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return R2D1 loss and Q-values."""

        (
            burnin_states,
            target_burnin_states,
            agent_states,
            target_states,
            burnin_prev_actions,
            target_burnin_prev_actions,
            agent_actions,
            prev_actions,
            target_prev_actions,
            burnin_prev_rewards,
            target_burnin_prev_rewards,
            agent_rewards,
            prev_rewards,
            target_prev_rewards,
            burnin_dones,
            burnin_target_dones,
            agent_dones,
            init_rnn_state,
        ) = slice_r2d2_arguments(experiences, head_cfg)

        with torch.no_grad():
            _, target_rnn_state = target_model(
                target_burnin_states,
                init_rnn_state,
                target_burnin_prev_actions,
                target_burnin_prev_rewards,
            )
            _, init_rnn_state = model(
                burnin_states, init_rnn_state, burnin_prev_actions, burnin_prev_rewards
            )

            init_rnn_state = torch.transpose(init_rnn_state, 0, 1)
            target_rnn_state = torch.transpose(target_rnn_state, 0, 1)

        burnin_invalid_mask = valid_from_done(burnin_dones.transpose(0, 1))
        burnin_target_invalid_mask = valid_from_done(
            burnin_target_dones.transpose(0, 1)
        )
        init_rnn_state[burnin_invalid_mask] = 0
        target_rnn_state[burnin_target_invalid_mask] = 0

        qs, _ = model(agent_states, init_rnn_state, prev_actions, prev_rewards)
        q = qs.gather(-1, agent_actions)
        with torch.no_grad():
            target_qs, _ = target_model(
                target_states,
                target_rnn_state,
                target_prev_actions,
                target_prev_rewards,
            )
            next_qs, _ = model(
                target_states, target_rnn_state, prev_actions, prev_rewards
            )
            next_a = torch.argmax(next_qs, dim=-1)
            target_q = target_qs.gather(-1, next_a.unsqueeze(-1))

        target = agent_rewards + gamma * target_q * (1 - agent_dones)
        dq_loss_element_wise = F.smooth_l1_loss(q, target.detach(), reduction="none")
        delta = abs(torch.mean(dq_loss_element_wise, dim=1))
        return delta, q


@LOSSES.register_module
class R2D1IQNLoss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return R2D1 loss and Q-values."""
        (
            burnin_states,
            target_burnin_states,
            agent_states,
            target_states,
            burnin_prev_actions,
            target_burnin_prev_actions,
            agent_actions,
            prev_actions,
            target_prev_actions,
            burnin_prev_rewards,
            target_burnin_prev_rewards,
            agent_rewards,
            prev_rewards,
            target_prev_rewards,
            burnin_dones,
            burnin_target_dones,
            agent_dones,
            init_rnn_state,
        ) = slice_r2d2_arguments(experiences, head_cfg)

        batch_size = agent_states.shape[0]
        sequence_size = agent_states.shape[1]

        with torch.no_grad():
            _, target_rnn_state = target_model(
                target_burnin_states,
                init_rnn_state,
                target_burnin_prev_actions,
                target_burnin_prev_rewards,
            )
            _, init_rnn_state = model(
                burnin_states, init_rnn_state, burnin_prev_actions, burnin_prev_rewards
            )

            init_rnn_state = torch.transpose(init_rnn_state, 0, 1)
            target_rnn_state = torch.transpose(target_rnn_state, 0, 1)

        burnin_invalid_mask = valid_from_done(burnin_dones.transpose(0, 1))
        burnin_target_invalid_mask = valid_from_done(
            burnin_target_dones.transpose(0, 1)
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
        qs, _ = model(
            target_states, target_rnn_state, target_prev_actions, target_prev_rewards
        )
        qs = qs.argmax(dim=-1)
        qs = qs[:, :, None]
        qs = qs.repeat(head_cfg.configs.n_tau_prime_samples, 1, 1)
        with torch.no_grad():
            # Shape of next_target_values: (n_tau_prime_samples x batch_size) x 1.
            target_quantile_values, _, _ = target_model.forward_(
                target_states,
                target_rnn_state,
                target_prev_actions,
                target_prev_rewards,
                head_cfg.configs.n_tau_prime_samples,
            )
        target_quantile_values = target_quantile_values.gather(-1, qs)
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
            agent_states,
            init_rnn_state,
            prev_actions,
            prev_rewards,
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
        quantiles = quantiles.to(device)

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
        q_values, _ = model(agent_states, init_rnn_state, prev_actions, prev_rewards)

        return iqn_loss_element_wise, q_values


@LOSSES.register_module
class R2D1C51Loss:
    def __call__(
        self,
        model: Brain,
        target_model: Brain,
        experiences: Tuple[torch.Tensor, ...],
        gamma: float,
        head_cfg: ConfigDict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return element-wise C51 loss and Q-values."""
        (
            burnin_states,
            target_burnin_states,
            agent_states,
            target_states,
            burnin_prev_actions,
            target_burnin_prev_actions,
            agent_actions,
            prev_actions,
            target_prev_actions,
            burnin_prev_rewards,
            target_burnin_prev_rewards,
            agent_rewards,
            prev_rewards,
            target_prev_rewards,
            burnin_dones,
            burnin_target_dones,
            agent_dones,
            init_rnn_state,
        ) = slice_r2d2_arguments(experiences, head_cfg)

        batch_size = agent_states.shape[0]
        sequence_size = agent_states.shape[1]

        with torch.no_grad():
            _, target_rnn_state = target_model(
                target_burnin_states,
                init_rnn_state,
                target_burnin_prev_actions,
                target_burnin_prev_rewards,
            )
            _, init_rnn_state = model(
                burnin_states, init_rnn_state, burnin_prev_actions, burnin_prev_rewards
            )

            init_rnn_state = torch.transpose(init_rnn_state, 0, 1)
            target_rnn_state = torch.transpose(target_rnn_state, 0, 1)

        burnin_invalid_mask = valid_from_done(burnin_dones.transpose(0, 1))
        burnin_target_invalid_mask = valid_from_done(
            burnin_target_dones.transpose(0, 1)
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
                target_states,
                target_rnn_state,
                target_prev_actions,
                target_prev_rewards,
            )
            next_actions = next_actions[1].argmax(-1)
            next_dist, _ = target_model.forward_(
                target_states,
                target_rnn_state,
                target_prev_actions,
                target_prev_rewards,
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
            agent_states, init_rnn_state, prev_actions, prev_rewards
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
