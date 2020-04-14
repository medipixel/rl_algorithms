# -*- coding: utf-8 -*-
"""DDPGfD agent using demo agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
         https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import pickle
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from rl_algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.ddpg.agent import DDPGAgent
from rl_algorithms.registry import AGENTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class DDPGfDAgent(DDPGAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (PrioritizedReplayBuffer): replay memory
        per_beta (float): beta parameter for prioritized replay buffer
        use_n_step (bool): whether or not to use n-step returns

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.per_beta = self.hyper_params.per_beta

        self.use_n_step = self.hyper_params.n_step > 1

        if not self.args.test:
            # load demo replay memory
            with open(self.args.demo_path, "rb") as f:
                demos = pickle.load(f)

            if self.use_n_step:
                demos, demos_n_step = common_utils.get_n_step_info_from_demo(
                    demos, self.hyper_params.n_step, self.hyper_params.gamma
                )

                # replay memory for multi-steps
                self.memory_n = ReplayBuffer(
                    buffer_size=self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                    demo=demos_n_step,
                )

            # replay memory for a single step
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                demo=demos,
                alpha=self.hyper_params.per_alpha,
                epsilon_d=self.hyper_params.per_eps_demo,
            )

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)

    def _get_critic_loss(
        self, experiences: Tuple[torch.Tensor, ...], gamma: float
    ) -> torch.Tensor:
        """Return element-wise critic loss."""
        states, actions, rewards, next_states, dones = experiences[:5]

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_states_actions = torch.cat((next_states, next_actions), dim=-1)
        next_values = self.critic_target(next_states_actions)
        curr_returns = rewards + gamma * next_values * masks
        curr_returns = curr_returns.to(device).detach()

        # train critic
        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss_element_wise = (values - curr_returns).pow(2)

        return critic_loss_element_wise

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        experiences_1 = self.memory.sample(self.per_beta)
        states, actions = experiences_1[:2]
        weights, indices, eps_d = experiences_1[-3:]
        gamma = self.hyper_params.gamma

        # train critic
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        critic_loss_element_wise = self._get_critic_loss(experiences_1, gamma)
        critic_loss = torch.mean(critic_loss_element_wise * weights)

        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = gamma ** self.hyper_params.n_step
            critic_loss_n_element_wise = self._get_critic_loss(experiences_n, gamma)
            # to update loss and priorities
            critic_loss_element_wise += (
                critic_loss_n_element_wise * self.hyper_params.lambda1
            )
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

        # update priorities
        new_priorities = critic_loss_element_wise
        new_priorities += self.hyper_params.lambda3 * actor_loss_element_wise.pow(2)
        new_priorities += self.hyper_params.per_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        new_priorities += eps_d
        self.memory.update_priorities(indices, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        return actor_loss.item(), critic_loss.item()

    def pretrain(self):
        """Pretraining steps."""
        pretrain_loss = list()
        pretrain_step = self.hyper_params.pretrain_step
        print("[INFO] Pre-Train %d step." % pretrain_step)
        for i_step in range(1, pretrain_step + 1):
            t_begin = time.time()
            loss = self.update_model()
            t_end = time.time()
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                log_value = (0, avg_loss, 0, t_end - t_begin)
                self.write_log(log_value)
        print("[INFO] Pre-Train Complete!\n")
