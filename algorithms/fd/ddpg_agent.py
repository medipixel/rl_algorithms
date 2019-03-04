# -*- coding: utf-8 -*-
"""DDPGfD agent using demo agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
         https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import pickle
from typing import Tuple

import numpy as np
import torch

from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBufferfD
import algorithms.common.helper_functions as common_utils
from algorithms.ddpg.agent import Agent as DDPGAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(DDPGAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (PrioritizedReplayBufferfD): replay memory
        beta (float): beta parameter for prioritized replay buffer

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # load demo replay memory
            with open(self.args.demo_path, "rb") as f:
                demo = pickle.load(f)

            # replay memory
            self.beta = self.hyper_params["PER_BETA"]
            self.memory = PrioritizedReplayBufferfD(
                self.hyper_params["BUFFER_SIZE"],
                self.hyper_params["BATCH_SIZE"],
                demo=list(demo),
                alpha=self.hyper_params["PER_ALPHA"],
                epsilon_d=self.hyper_params["PER_EPS_DEMO"],
            )

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        experiences = self.memory.sample(self.beta)
        states, actions, rewards, next_states, dones, weights, indexes, eps_d = (
            experiences
        )
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_states_actions = torch.cat((next_states, next_actions), dim=-1)
        next_values = self.critic_target(next_states_actions)
        curr_returns = rewards + self.hyper_params["GAMMA"] * next_values * masks
        curr_returns = curr_returns.to(device).detach()

        # train critic
        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss_element_wise = (values - curr_returns).pow(2)
        critic_loss = torch.mean(critic_loss_element_wise * weights)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        actions = self.actor(states)
        actor_loss_element_wise = -self.critic(torch.cat((states, actions), dim=-1))
        actor_loss = torch.mean(actor_loss_element_wise * weights)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        tau = self.hyper_params["TAU"]
        common_utils.soft_update(self.actor, self.actor_target, tau)
        common_utils.soft_update(self.critic, self.critic_target, tau)

        # update priorities
        new_priorities = critic_loss_element_wise
        new_priorities += self.hyper_params["LAMBDA3"] * actor_loss_element_wise.pow(2)
        new_priorities += self.hyper_params["PER_EPS"]
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        new_priorities += eps_d
        self.memory.update_priorities(indexes, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        return actor_loss.data, critic_loss.data

    def pretrain(self):
        """Pretraining steps."""
        pretrain_loss = list()
        print("[INFO] Pre-Train %d step." % self.hyper_params["PRETRAIN_STEP"])
        for i_step in range(1, self.hyper_params["PRETRAIN_STEP"] + 1):
            loss = self.update_model()
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                self.write_log(0, avg_loss, 0)
