# -*- coding: utf-8 -*-
"""SAC agent from demonstration for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
         https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import pickle
import time
from typing import Tuple

import numpy as np
import torch

from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBufferfD
from algorithms.common.buffer.replay_buffer import NStepTransitionBuffer
import algorithms.common.helper_functions as common_utils
from algorithms.sac.agent import SACAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SACfDAgent(SACAgent):
    """SAC agent interacting with environment.

    Attrtibutes:
        memory (PrioritizedReplayBufferfD): replay memory
        beta (float): beta parameter for prioritized replay buffer

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.use_n_step = self.hyper_params["N_STEP"] > 1

        if not self.args.test:
            # load demo replay memory
            with open(self.args.demo_path, "rb") as f:
                demos = pickle.load(f)

            if self.use_n_step:
                demos, demos_n_step = common_utils.get_n_step_info_from_demo(
                    demos, self.hyper_params["N_STEP"], self.hyper_params["GAMMA"]
                )

                # replay memory for multi-steps
                self.memory_n = NStepTransitionBuffer(
                    buffer_size=self.hyper_params["BUFFER_SIZE"],
                    n_step=self.hyper_params["N_STEP"],
                    gamma=self.hyper_params["GAMMA"],
                    demo=demos_n_step,
                )

            # replay memory
            self.beta = self.hyper_params["PER_BETA"]
            self.memory = PrioritizedReplayBufferfD(
                self.hyper_params["BUFFER_SIZE"],
                self.hyper_params["BATCH_SIZE"],
                demo=demos,
                alpha=self.hyper_params["PER_ALPHA"],
                epsilon_d=self.hyper_params["PER_EPS_DEMO"],
            )

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(*transition)

    # pylint: disable=too-many-statements
    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        self.update_step += 1

        experiences = self.memory.sample(self.beta)
        states, actions, rewards, next_states, dones, weights, indices, eps_d = (
            experiences
        )
        new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)

        # train alpha
        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            alpha_loss = torch.mean(
                (-self.log_alpha * (log_prob + self.target_entropy).detach()) * weights
            )

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params["W_ENTROPY"]

        # Q function loss
        masks = 1 - dones
        gamma = self.hyper_params["GAMMA"]
        q_1_pred = self.qf_1(states, actions)
        q_2_pred = self.qf_2(states, actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.hyper_params["GAMMA"] * v_target * masks
        qf_1_loss = torch.mean((q_1_pred - q_target.detach()).pow(2) * weights)
        qf_2_loss = torch.mean((q_2_pred - q_target.detach()).pow(2) * weights)

        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            _, _, rewards, next_states, dones = experiences_n
            gamma = gamma ** self.hyper_params["N_STEP"]
            lambda1 = self.hyper_params["LAMBDA1"]
            masks = 1 - dones

            v_target = self.vf_target(next_states)
            q_target = rewards + gamma * v_target * masks
            qf_1_loss_n = torch.mean((q_1_pred - q_target.detach()).pow(2) * weights)
            qf_2_loss_n = torch.mean((q_2_pred - q_target.detach()).pow(2) * weights)

            # to update loss and priorities
            qf_1_loss = qf_1_loss + qf_1_loss_n * lambda1
            qf_2_loss = qf_2_loss + qf_2_loss_n * lambda1

        # V function loss
        v_pred = self.vf(states)
        q_pred = torch.min(
            self.qf_1(states, new_actions), self.qf_2(states, new_actions)
        )
        v_target = (q_pred - alpha * log_prob).detach()
        vf_loss_element_wise = (v_pred - v_target).pow(2)
        vf_loss = torch.mean(vf_loss_element_wise * weights)

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
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss_element_wise = alpha * log_prob - advantage
            actor_loss = torch.mean(actor_loss_element_wise * weights)

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

            # update priorities
            new_priorities = vf_loss_element_wise
            new_priorities += self.hyper_params[
                "LAMBDA3"
            ] * actor_loss_element_wise.pow(2)
            new_priorities += self.hyper_params["PER_EPS"]
            new_priorities = new_priorities.data.cpu().numpy().squeeze()
            new_priorities += eps_d
            self.memory.update_priorities(indices, new_priorities)

            # increase beta
            fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)
        else:
            actor_loss = torch.zeros(1)

        return (
            actor_loss.item(),
            qf_1_loss.item(),
            qf_2_loss.item(),
            vf_loss.item(),
            alpha_loss.item(),
        )

    def pretrain(self):
        """Pretraining steps."""
        pretrain_loss = list()
        print("[INFO] Pre-Train %d steps." % self.hyper_params["PRETRAIN_STEP"])
        for i_step in range(1, self.hyper_params["PRETRAIN_STEP"] + 1):
            t_begin = time.time()
            loss = self.update_model()
            t_end = time.time()
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                self.write_log(
                    0,
                    avg_loss,
                    0,
                    policy_update_freq=self.hyper_params["POLICY_UPDATE_FREQ"],
                    avg_time_cost=t_end - t_begin,
                )
        print("[INFO] Pre-Train Complete!\n")
