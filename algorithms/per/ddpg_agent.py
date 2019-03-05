# -*- coding: utf-8 -*-
"""DDPG agent with PER for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
         https://arxiv.org/pdf/1511.05952.pdf
"""

from typing import Tuple

import torch

from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
import algorithms.common.helper_functions as common_utils
from algorithms.ddpg.agent import Agent as DDPGAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(DDPGAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (PrioritizedReplayBuffer): replay memory
        beta (float): beta parameter for prioritized replay buffer

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory
            self.beta = self.hyper_params["PER_BETA"]
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params["BUFFER_SIZE"],
                self.hyper_params["BATCH_SIZE"],
                alpha=self.hyper_params["PER_ALPHA"],
            )

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        experiences = self.memory.sample(self.beta)
        states, actions, rewards, next_states, dones, weights, indexes = experiences

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
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

        # update priorities in PER
        new_priorities = critic_loss_element_wise
        new_priorities = (
            new_priorities.data.cpu().numpy() + self.hyper_params["PER_EPS"]
        )
        self.memory.update_priorities(indexes, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        return actor_loss.data, critic_loss.data
