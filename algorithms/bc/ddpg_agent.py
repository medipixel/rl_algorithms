# -*- coding: utf-8 -*-
"""DDPG with Behavior Cloning agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1709.10089.pdf
"""

import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from algorithms.common.buffer.replay_buffer import ReplayBuffer
import algorithms.common.helper_functions as common_utils
from algorithms.ddpg.agent import Agent as DDPGAgent
from algorithms.her import HER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(DDPGAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (ReplayBuffer): replay memory
        demo_memory (ReplayBuffer): replay memory for demo
        her (HER): hinsight experience replay
        transitions_epi (list): transitions per episode (for HER)
        goal_state (np.ndarray): goal state to generate concatenated states
        total_step (int): total step numbers
        episode_step (int): step number of the current episode

    """

    # pylint: disable=attribute-defined-outside-init
    def _init_replay_buffer(self):
        """Initialize replay buffer."""
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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        if self.hyper_params["USE_HER"]:
            self.desired_state = self.her.sample_desired_state()
            state = np.concatenate((state, self.desired_state), axis=-1)

        selected_action = DDPGAgent.select_action(self, state)
        self.curr_state = state

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done = DDPGAgent.step(self, action)

        if not self.args.test and self.hyper_params["USE_HER"]:
            self.transitions_epi.append(self.hooked_transition)
            if done:
                # insert generated transitions if the episode is done
                transitions = self.her.generate_transitions(
                    self.transitions_epi, self.desired_state
                )
                self.memory.extend(transitions)

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
        curr_returns = rewards + (self.hyper_params["GAMMA"] * next_values * masks)
        curr_returns = curr_returns.to(device)

        # critic loss
        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)

        # train critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

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
        actor_loss = self.lambda1 * policy_loss + self.lambda2 * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        tau = self.hyper_params["TAU"]
        common_utils.soft_update(self.actor, self.actor_target, tau)
        common_utils.soft_update(self.critic, self.critic_target, tau)

        return actor_loss.data, critic_loss.data

    def write_log(self, i_episode: int, loss: np.ndarray, score: float = 0.0):
        """Write log about loss and score"""
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "total loss: %f actor_loss: %.3f critic_loss: %.3f\n"
            % (
                i_episode,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0],
                loss[1],
            )  # actor loss  # critic loss
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0],
                    "critic loss": loss[1],
                }
            )
