# -*- coding: utf-8 -*-
"""BC with SAC agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
         https://arxiv.org/pdf/1709.10089.pdf
"""

import argparse
import pickle
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from algorithms.common.abstract.her import HER
from algorithms.common.buffer.replay_buffer import ReplayBuffer
import algorithms.common.helper_functions as common_utils
from algorithms.registry import AGENTS, build_her
from algorithms.sac.agent import SACAgent
from algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class BCSACAgent(SACAgent):
    """BC with SAC agent interacting with environment.

    Attrtibutes:
        her (HER): hinsight experience replay
        transitions_epi (list): transitions per episode (for HER)
        desired_state (np.ndarray): desired state of current episode
        memory (ReplayBuffer): replay memory
        demo_memory (ReplayBuffer): replay memory for demo
        lambda1 (float): proportion of policy loss
        lambda2 (float): proportion of BC loss

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.demo_batch_size = self.params.demo_batch_size
        self.lambda1 = self.params.lambda1
        self.lambda2 = self.params.lambda2
        self.use_her = self.params.use_her
        self.success_score = self.params.success_score
        self.desired_states_from_demo = self.params.desired_states_from_demo

        # load demo replay memory
        with open(self.args.demo_path, "rb") as f:
            demo = list(pickle.load(f))

        # HER
        if self.use_her:
            self.her = build_her(self.params.her)
            print(f"[INFO] Build {str(self.her)}.")

            if self.desired_states_from_demo:
                self.her.fetch_desired_states_from_demo(demo)

            self.transitions_epi: list = list()
            self.desired_state = np.zeros((1,))
            demo = self.her.generate_demo_transitions(demo)

            if not self.her.is_goal_in_state:
                self.state_dim *= 2
        else:
            self.her = None

        self._init_network()

        if not self.args.test:
            # Replay buffers
            self.demo_memory = ReplayBuffer(len(demo), self.demo_batch_size)
            self.demo_memory.extend(demo)

            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

            # set hyper parameters
            self.lambda1 = self.lambda1
            self.lambda2 = self.lambda2 / self.demo_batch_size

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        if self.use_her:
            self.desired_state = self.her.get_desired_state()
            state = np.concatenate((state, self.desired_state), axis=-1)
        state = torch.FloatTensor(state).to(device)
        return state

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        if self.use_her:
            self.transitions_epi.append(transition)
            done = transition[-1] or self.episode_step == self.args.max_episode_steps
            if done:
                # insert generated transitions if the episode is done
                transitions = self.her.generate_transitions(
                    self.transitions_epi, self.desired_state, self.success_score,
                )
                self.memory.extend(transitions)
                self.transitions_epi.clear()
        else:
            self.memory.add(transition)

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        self.update_step += 1

        experiences, demos = self.memory.sample(), self.demo_memory.sample()

        states, actions, rewards, next_states, dones = experiences
        demo_states, demo_actions, _, _, _ = demos
        new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)
        pred_actions, _, _, _, _ = self.actor(demo_states)

        # train alpha
        if self.auto_entropy_tuning:
            alpha_loss = (
                -self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.w_entropy

        # Q function loss
        masks = 1 - dones
        q_1_pred = self.qf_1(states, actions)
        q_2_pred = self.qf_2(states, actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.gamma * v_target * masks
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        # V function loss
        v_pred = self.vf(states)
        q_pred = torch.min(
            self.qf_1(states, new_actions), self.qf_2(states, new_actions)
        )
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        # train Q functions
        self.qf_1_optim.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optim.step()

        self.qf_2_optim.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optim.step()

        # train V function
        self.vf_optim.zero_grad()
        vf_loss.backward()
        self.vf_optim.step()

        if self.update_step % self.policy_update_freq == 0:
            # bc loss
            qf_mask = torch.gt(
                self.qf_1(demo_states, demo_actions),
                self.qf_1(demo_states, pred_actions),
            ).to(device)
            qf_mask = qf_mask.float()
            n_qf_mask = int(qf_mask.sum().item())

            if n_qf_mask == 0:
                bc_loss = torch.zeros(1, device=device)
            else:
                bc_loss = (
                    torch.mul(pred_actions, qf_mask) - torch.mul(demo_actions, qf_mask)
                ).pow(2).sum() / n_qf_mask

            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()
            actor_loss = self.lambda1 * actor_loss + self.lambda2 * bc_loss

            # regularization
            if not self.is_discrete:  # iff the action is continuous
                mean_reg = self.w_mean_reg * mu.pow(2).mean()
                std_reg = self.w_std_reg * std.pow(2).mean()
                pre_activation_reg = self.w_pre_activation_reg * (
                    pre_tanh_value.pow(2).sum(dim=-1).mean()
                )
                actor_reg = mean_reg + std_reg + pre_activation_reg

                # actor loss + regularization
                actor_loss += actor_reg

            # train actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            common_utils.soft_update(self.vf, self.vf_target, self.tau)
        else:
            actor_loss = torch.zeros(1)
            n_qf_mask = 0

        return (
            actor_loss.item(),
            qf_1_loss.item(),
            qf_2_loss.item(),
            vf_loss.item(),
            alpha_loss.item(),
            n_qf_mask,
        )

    def write_log(
        self,
        i: int,
        loss: np.ndarray,
        score: float = 0.0,
        policy_update_freq: int = 1,
        avg_time_cost: float = 0.0,
    ):
        """Write log about loss and score"""
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "total loss: %.3f actor_loss: %.3f qf_1_loss: %.3f qf_2_loss: %.3f "
            "vf_loss: %.3f alpha_loss: %.3f n_qf_mask: %d (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0] * policy_update_freq,  # actor loss
                loss[1],  # qf_1 loss
                loss[2],  # qf_2 loss
                loss[3],  # vf loss
                loss[4],  # alpha loss
                loss[5],  # n_qf_mask
                avg_time_cost,
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * policy_update_freq,
                    "qf_1 loss": loss[1],
                    "qf_2 loss": loss[2],
                    "vf loss": loss[3],
                    "alpha loss": loss[4],
                    "time per each step": avg_time_cost,
                }
            )
