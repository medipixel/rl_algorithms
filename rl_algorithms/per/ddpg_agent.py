# -*- coding: utf-8 -*-
"""DDPG agent with PER for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
         https://arxiv.org/pdf/1511.05952.pdf
"""

from typing import Tuple

from rl_algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.ddpg.agent import DDPGAgent
from rl_algorithms.registry import AGENTS
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class PERDDPGAgent(DDPGAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (PrioritizedReplayBuffer): replay memory
        per_beta (float): beta parameter for prioritized replay buffer

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.per_beta = self.hyper_params.per_beta

        if not self.args.test:
            # replay memory
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                alpha=self.hyper_params.per_alpha,
            )

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        experiences = self.memory.sample(self.per_beta)
        states, actions, rewards, next_states, dones, weights, indices, _ = experiences

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + self.hyper_params.gamma * next_values * masks
        curr_returns = curr_returns.to(device).detach()

        # train critic
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss_element_wise = (values - curr_returns).pow(2)
        critic_loss = torch.mean(critic_loss_element_wise * weights)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        # train actor
        actions = self.actor(states)
        actor_loss_element_wise = -self.critic(torch.cat((states, actions), dim=-1))
        actor_loss = torch.mean(actor_loss_element_wise * weights)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        # update target networks
        common_utils.soft_update(self.actor, self.actor_target, self.hyper_params.tau)
        common_utils.soft_update(self.critic, self.critic_target, self.hyper_params.tau)

        # update priorities in PER
        new_priorities = critic_loss_element_wise
        new_priorities = new_priorities.data.cpu().numpy() + self.hyper_params.per_eps
        self.memory.update_priorities(indices, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        return actor_loss.item(), critic_loss.item()
