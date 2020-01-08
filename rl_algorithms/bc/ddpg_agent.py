# -*- coding: utf-8 -*-
"""Behaviour Cloning with DDPG agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1709.10089.pdf
"""

import pickle
from typing import Tuple

import numpy as np
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.ddpg.agent import DDPGAgent
from rl_algorithms.registry import AGENTS, build_her
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class BCDDPGAgent(DDPGAgent):
    """BC with DDPG agent interacting with environment.

    Attributes:
        her (HER): hinsight experience replay
        transitions_epi (list): transitions per episode (for HER)
        desired_state (np.ndarray): desired state of current episode
        memory (ReplayBuffer): replay memory
        demo_memory (ReplayBuffer): replay memory for demo
        lambda2 (float): proportion of BC loss

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        # load demo replay memory
        with open(self.args.demo_path, "rb") as f:
            demo = list(pickle.load(f))

        # HER
        if self.hyper_params.use_her:
            self.her = build_her(self.hyper_params.her)
            print(f"[INFO] Build {str(self.her)}.")

            if self.hyper_params.desired_states_from_demo:
                self.her.fetch_desired_states_from_demo(demo)

            self.transitions_epi: list = list()
            self.desired_state = np.zeros((1,))
            demo = self.her.generate_demo_transitions(demo)

            if not self.her.is_goal_in_state:
                self.state_dim *= 2
        else:
            self.her = None

        if not self.args.test:
            # Replay buffers
            demo_batch_size = self.hyper_params.demo_batch_size
            self.demo_memory = ReplayBuffer(len(demo), demo_batch_size)
            self.demo_memory.extend(demo)

            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size
            )

            # set hyper parameters
            self.lambda2 = 1.0 / demo_batch_size

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        if self.hyper_params.use_her:
            self.desired_state = self.her.get_desired_state()
            state = np.concatenate((state, self.desired_state), axis=-1)
        state = torch.FloatTensor(state).to(device)
        return state

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        if self.hyper_params.use_her:
            self.transitions_epi.append(transition)
            done = transition[-1] or self.episode_step == self.args.max_episode_steps
            if done:
                # insert generated transitions if the episode is done
                transitions = self.her.generate_transitions(
                    self.transitions_epi,
                    self.desired_state,
                    self.hyper_params.success_score,
                )
                self.memory.extend(transitions)
                self.transitions_epi.clear()
        else:
            self.memory.add(transition)

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        experiences = self.memory.sample()
        demos = self.demo_memory.sample()
        exp_states, exp_actions, exp_rewards, exp_next_states, exp_dones = experiences
        demo_states, demo_actions, demo_rewards, demo_next_states, demo_dones = demos

        states = torch.cat((exp_states, demo_states), dim=0)
        actions = torch.cat((exp_actions, demo_actions), dim=0)
        rewards = torch.cat((exp_rewards, demo_rewards), dim=0)
        next_states = torch.cat((exp_next_states, demo_next_states), dim=0)
        dones = torch.cat((exp_dones, demo_dones), dim=0)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + (self.hyper_params.gamma * next_values * masks)
        curr_returns = curr_returns.to(device)

        # critic loss
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)

        # train critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        # policy loss
        actions = self.actor(states)
        policy_loss = -self.critic(torch.cat((states, actions), dim=-1)).mean()

        # bc loss
        pred_actions = self.actor(demo_states)
        qf_mask = torch.gt(
            self.critic(torch.cat((demo_states, demo_actions), dim=-1)),
            self.critic(torch.cat((demo_states, pred_actions), dim=-1)),
        ).to(device)
        qf_mask = qf_mask.float()
        n_qf_mask = int(qf_mask.sum().item())

        if n_qf_mask == 0:
            bc_loss = torch.zeros(1, device=device)
        else:
            bc_loss = (
                torch.mul(pred_actions, qf_mask) - torch.mul(demo_actions, qf_mask)
            ).pow(2).sum() / n_qf_mask

        # train actor: pg loss + BC loss
        actor_loss = self.hyper_params.lambda1 * policy_loss + self.lambda2 * bc_loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        # update target networks
        common_utils.soft_update(self.actor, self.actor_target, self.hyper_params.tau)
        common_utils.soft_update(self.critic, self.critic_target, self.hyper_params.tau)

        return actor_loss.item(), critic_loss.item(), n_qf_mask

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, avg_time_cost = log_value
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "total loss: %f actor_loss: %.3f critic_loss: %.3f, n_qf_mask: %d "
            "(spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0],
                loss[1],
                loss[2],
                avg_time_cost,
            )  # actor loss  # critic loss
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0],
                    "critic loss": loss[1],
                    "time per each step": avg_time_cost,
                }
            )
