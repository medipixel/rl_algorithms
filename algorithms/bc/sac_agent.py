# -*- coding: utf-8 -*-
"""BC with SAC agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
         https://arxiv.org/pdf/1709.10089.pdf
"""

import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.common.buffer.replay_buffer import ReplayBuffer
import algorithms.common.helper_functions as common_utils
from algorithms.her import HER
from algorithms.sac.agent import Agent as SACAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(SACAgent):
    """SAC agent interacting with environment.

    Attrtibutes:
        memory (ReplayBuffer): replay memory
        demo_memory (ReplayBuffer): replay memory for demo
        her (HER): hinsight experience replay
        transitions_epi (list): transitions per episode (for HER)
        goal_state (np.ndarray): goal state to generate concatenated states
        total_step (int): total step numbers
        episode_step (int): step number of the current episode

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        # load demo replay memory
        with open(self.args.demo_path, "rb") as f:
            demo = list(pickle.load(f))

        # HER
        if self.hyper_params["USE_HER"]:
            self.her = HER(self.args.demo_path)
            self.transitions_epi: list = list()
            self.desired_state = np.zeros((1,))
            self.hook_transition = True
            demo = self.her.generate_demo_transitions(demo)

        if not self.args.test:
            # Replay buffers
            self.demo_memory = ReplayBuffer(
                len(demo), self.hyper_params["DEMO_BATCH_SIZE"]
            )
            self.demo_memory.extend(demo)

            self.memory = ReplayBuffer(
                self.hyper_params["BUFFER_SIZE"], self.hyper_params["BATCH_SIZE"]
            )

            # set hyper parameters
            self.lambda1 = self.hyper_params["LAMBDA1"]
            self.lambda2 = (
                self.hyper_params["LAMBDA2"] / self.hyper_params["DEMO_BATCH_SIZE"]
            )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        state_ = state

        # HER
        if self.hyper_params["USE_HER"]:
            self.desired_state = self.her.sample_desired_state()
            state = np.concatenate((state, self.desired_state), axis=-1)

        selected_action = SACAgent.select_action(self, state)
        self.curr_state = state_

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done = SACAgent.step(self, action)

        if not self.args.test and self.hyper_params["USE_HER"]:
            self.transitions_epi.append(self.hooked_transition)
            if done:
                # insert generated transitions if the episode is done
                transitions = self.her.generate_transitions(
                    self.transitions_epi, self.desired_state
                )
                self.memory.extend(transitions)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        experiences = self.memory.sample()
        demos = self.demo_memory.sample()

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

        if self.total_step % self.hyper_params["DELAYED_UPDATE"] == 0:
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

        return (
            actor_loss.data,
            qf_1_loss.data,
            qf_2_loss.data,
            vf_loss.data,
            alpha_loss.data,
        )
