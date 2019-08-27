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
from algorithms.sac.agent import SACAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
        target_entropy: float,
        her: HER,
    ):
        """Initialization.
        Args:
            her (HER): hinsight experience replay

        """
        self.her = her
        SACAgent.__init__(self, env, args, hyper_params, models, optims, target_entropy)

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        # load demo replay memory
        with open(self.args.demo_path, "rb") as f:
            demo = list(pickle.load(f))

        # HER
        if self.hyper_params["USE_HER"]:
            if self.hyper_params["DESIRED_STATES_FROM_DEMO"]:
                self.her.fetch_desired_states_from_demo(demo)

            self.transitions_epi: list = list()
            self.desired_state = np.zeros((1,))
            demo = self.her.generate_demo_transitions(demo)

        if not self.args.test:
            # Replay buffers
            demo_batch_size = self.hyper_params["DEMO_BATCH_SIZE"]
            self.demo_memory = ReplayBuffer(len(demo), demo_batch_size)
            self.demo_memory.extend(demo)

            self.memory = ReplayBuffer(
                self.hyper_params["BUFFER_SIZE"], self.hyper_params["BATCH_SIZE"]
            )

            # set hyper parameters
            self.lambda1 = self.hyper_params["LAMBDA1"]
            self.lambda2 = self.hyper_params["LAMBDA2"] / demo_batch_size

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        if self.hyper_params["USE_HER"]:
            self.desired_state = self.her.get_desired_state()
            state = np.concatenate((state, self.desired_state), axis=-1)
        state = torch.FloatTensor(state).to(device)
        return state

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        if self.hyper_params["USE_HER"]:
            self.transitions_epi.append(transition)
            done = transition[-1] or self.episode_step == self.args.max_episode_steps
            if done:
                # insert generated transitions if the episode is done
                transitions = self.her.generate_transitions(
                    self.transitions_epi,
                    self.desired_state,
                    self.hyper_params["SUCCESS_SCORE"],
                )
                self.memory.extend(transitions)
                self.transitions_epi.clear()
        else:
            self.memory.add(*transition)

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        self.update_step += 1

        experiences, demos = self.memory.sample(), self.demo_memory.sample()

        states, actions, rewards, next_states, dones = experiences
        demo_states, demo_actions, _, _, _ = demos
        new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)
        pred_actions, _, _, _, _ = self.actor(demo_states)

        # train alpha
        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            alpha_loss = (
                -self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params["W_ENTROPY"]

        # Q function loss
        masks = 1 - dones
        q_1_pred = self.qf_1(states, actions)
        q_2_pred = self.qf_2(states, actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.hyper_params["GAMMA"] * v_target * masks
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
        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        if self.update_step % self.hyper_params["POLICY_UPDATE_FREQ"] == 0:
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
                mean_reg = self.hyper_params["W_MEAN_REG"] * mu.pow(2).mean()
                std_reg = self.hyper_params["W_STD_REG"] * std.pow(2).mean()
                pre_activation_reg = self.hyper_params["W_PRE_ACTIVATION_REG"] * (
                    pre_tanh_value.pow(2).sum(dim=-1).mean()
                )
                actor_reg = mean_reg + std_reg + pre_activation_reg

                # actor loss + regularization
                actor_loss += actor_reg

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            common_utils.soft_update(self.vf, self.vf_target, self.hyper_params["TAU"])
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
