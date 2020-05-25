# -*- coding: utf-8 -*-
"""R2D1 agent, which implement R2D2 but without distributed actor.
- Author: Kyunghwan Kim, Curt Park, Euijin Jeong
- Contact:kh.kim@medipixel.io, curt.park@medipixel.io, euijin.jeong@medipixel.io
- Paper: https://openreview.net/pdf?id=r1lyTjAqYX (R2D1)
"""

import time
from typing import Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import wandb

from rl_algorithms.common.buffer.recurrent_prioritized_replay_buffer import (
    RecurrentPrioritizedReplayBuffer,
)
from rl_algorithms.common.buffer.recurrent_replay_buffer import RecurrentReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.brain import GRUBrain
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.registry import AGENTS, build_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class R2D1Agent(DQNAgent):
    """R2D1 interacting with environment.

    Attribute:
        memory (RecurrentPrioritizedReplayBuffer): replay memory for recurrent agent
        memory_n (RecurrentReplayBuffer): nstep replay memory for recurrent agent
    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:

            self.memory = RecurrentPrioritizedReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                self.hyper_params.sequence_size,
                self.hyper_params.overlap_size,
                alpha=self.hyper_params.per_alpha,
            )

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = RecurrentReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    self.hyper_params.sequence_size,
                    self.hyper_params.overlap_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

    def _init_network(self):
        """Initialize networks and optimizers."""

        self.head_cfg.configs.state_size = self.state_dim
        self.head_cfg.configs.output_size = self.action_dim

        self.dqn = GRUBrain(self.backbone_cfg, self.head_cfg).to(device)
        self.dqn_target = GRUBrain(self.backbone_cfg, self.head_cfg).to(device)
        self.loss_fn = build_loss(self.hyper_params.loss_type)

        self.dqn_target.load_state_dict(self.dqn.state_dict())

        # create optimizer
        self.dqn_optim = optim.Adam(
            self.dqn.parameters(),
            lr=self.optim_cfg.lr_dqn,
            weight_decay=self.optim_cfg.weight_decay,
            eps=self.optim_cfg.adam_eps,
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def select_action(
        self,
        state: np.ndarray,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: np.ndarray,
    ) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        # pylint: disable=comparison-with-callable
        state = self._preprocess_state(state)
        selected_action, hidden_state = self.dqn(
            state, hidden_state, prev_action, prev_reward
        )
        selected_action = selected_action.detach().argmax().cpu().numpy()
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        return selected_action, hidden_state

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(transition)

    def step(
        self, action: np.ndarray, hidden_state: torch.Tensor
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)
        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )

            transition = (
                self.curr_state,
                action,
                hidden_state.detach(),
                reward,
                next_state,
                done_bool,
            )
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        # 1 step loss
        experiences_1 = self.memory.sample(self.per_beta)
        weights, indices = experiences_1[-3:-1]
        gamma = self.hyper_params.gamma
        dq_loss_element_wise, q_values = self.loss_fn(
            self.dqn, self.dqn_target, experiences_1, gamma, self.head_cfg
        )
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self.loss_fn(
                self.dqn, self.dqn_target, experiences_n, gamma, self.head_cfg
            )

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params.w_q_reg

        # total loss
        loss = dq_loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward(retain_graph=True)
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params.per_eps
        self.memory.update_priorities(indices, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

        if self.head_cfg.configs.use_noisy_net:
            self.dqn.head.reset_noise()
            self.dqn_target.head.reset_noise()

        return loss.item(), q_values.mean().item()

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.dqn], log="parameters")

        # pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            hidden_in = torch.zeros(
                [1, 1, self.head_cfg.configs.rnn_hidden_size], dtype=torch.float
            ).to(device)
            prev_action = torch.zeros(1, 1, self.action_dim).to(device)
            prev_reward = torch.zeros(1, 1, 1).to(device)
            self.episode_step = 0
            self.sequence_step = 0
            losses = list()
            done = False
            score = 0

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action, hidden_out = self.select_action(
                    state, hidden_in, prev_action, prev_reward
                )
                next_state, reward, done, _ = self.step(action, hidden_in)
                self.total_step += 1
                self.episode_step += 1

                if self.episode_step % self.hyper_params.sequence_size == 0:
                    self.sequence_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:

                    if self.sequence_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            loss = self.update_model()
                            losses.append(loss)  # for logging

                    # decrease epsilon
                    self.epsilon = max(
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon)
                        * self.hyper_params.epsilon_decay,
                        self.min_epsilon,
                    )
                hidden_in = hidden_out
                state = next_state
                prev_action = common_utils.make_one_hot(
                    torch.as_tensor(action), self.action_dim
                )
                prev_reward = torch.as_tensor(reward).to(device)
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)

                if self.i_episode % self.args.save_period == 0:
                    self.save_params(self.i_episode)
                    self.interim_test()

        # termination
        self.env.close()
        self.save_params(self.i_episode)
        self.interim_test()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.args.interim_test_num
        else:
            test_num = self.args.episode_num
        score_list = []
        for i_episode in range(test_num):
            hidden_in = torch.zeros(
                [1, 1, self.head_cfg.configs.rnn_hidden_size], dtype=torch.float
            ).to(device)
            prev_action = torch.zeros(1, 1, self.action_dim).to(device)
            prev_reward = torch.zeros(1, 1, 1).to(device)
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.args.render:
                    self.env.render()

                action, hidden_out = self.select_action(
                    state, hidden_in, prev_action, prev_reward
                )
                next_state, reward, done, _ = self.step(action, hidden_in)

                hidden_in = hidden_out
                state = next_state
                prev_action = common_utils.make_one_hot(
                    torch.as_tensor(action), self.action_dim
                )
                prev_reward = torch.as_tensor(reward).to(device)
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            score_list.append(score)

        if self.args.log:
            wandb.log(
                {
                    "test score": round(sum(score_list) / len(score_list), 2),
                    "test total step": self.total_step,
                }
            )
