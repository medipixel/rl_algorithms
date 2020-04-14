# -*- coding: utf-8 -*-
"""DQfD agent using demo agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim, Curt Park
- Contact: kh.kim@medipixel.io, curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1704.03732.pdf (DQfD)
"""

import pickle
import time
from typing import Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import wandb

from rl_algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBuffer
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.registry import AGENTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class DQfDAgent(DQNAgent):
    """DQN interacting with environment.

    Attribute:
        memory (PrioritizedReplayBuffer): replay memory

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # load demo replay memory
            demos = self._load_demos()

            if self.use_n_step:
                demos, demos_n_step = common_utils.get_n_step_info_from_demo(
                    demos, self.hyper_params.n_step, self.hyper_params.gamma
                )

                self.memory_n = ReplayBuffer(
                    buffer_size=self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                    demo=demos_n_step,
                )

            # replay memory
            self.memory = PrioritizedReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                demo=demos,
                alpha=self.hyper_params.per_alpha,
                epsilon_d=self.hyper_params.per_eps_demo,
            )

    def _load_demos(self) -> list:
        """Load expert's demonstrations."""
        # load demo replay memory
        with open(self.args.demo_path, "rb") as f:
            demos = pickle.load(f)

        return demos

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        experiences_1 = self.memory.sample()
        weights, indices, eps_d = experiences_1[-3:]
        actions = experiences_1[1]

        # 1 step loss
        gamma = self.hyper_params.gamma
        dq_loss_element_wise, q_values = self._get_dqn_loss(experiences_1, gamma)
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step
            dq_loss_n_element_wise, q_values_n = self._get_dqn_loss(
                experiences_n, gamma
            )

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.lambda1
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # supervised loss using demo for only demo transitions
        demo_idxs = np.where(eps_d != 0.0)
        n_demo = demo_idxs[0].size
        if n_demo != 0:  # if 1 or more demos are sampled
            # get margin for each demo transition
            action_idxs = actions[demo_idxs].long()
            margin = torch.ones(q_values.size()) * self.hyper_params.margin
            margin[demo_idxs, action_idxs] = 0.0  # demo actions have 0 margins
            margin = margin.to(device)

            # calculate supervised loss
            demo_q_values = q_values[demo_idxs, action_idxs].squeeze()
            supervised_loss = torch.max(q_values + margin, dim=-1)[0]
            supervised_loss = supervised_loss[demo_idxs] - demo_q_values
            supervised_loss = torch.mean(supervised_loss) * self.hyper_params.lambda2
        else:  # no demo sampled
            supervised_loss = torch.zeros(1, device=device)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params.w_q_reg

        # total loss
        loss = dq_loss + supervised_loss + q_regular

        # train dqn
        self.dqn_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        common_utils.soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy().squeeze()
        new_priorities = loss_for_prior + self.hyper_params.per_eps
        new_priorities += eps_d
        self.memory.update_priorities(indices, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
        self.per_beta: float = self.per_beta + fraction * (1.0 - self.per_beta)

        if self.hyper_params.use_noisy_net:
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

        return (
            loss.item(),
            dq_loss.item(),
            supervised_loss.item(),
            q_values.mean().item(),
            n_demo,
        )

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, avg_loss, score, avg_time_cost = log_value
        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %f\n"
            "epsilon: %f, total loss: %f, dq loss: %f, supervised loss: %f\n"
            "avg q values: %f, demo num in minibatch: %d (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                self.epsilon,
                avg_loss[0],
                avg_loss[1],
                avg_loss[2],
                avg_loss[3],
                avg_loss[4],
                avg_time_cost,
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "epsilon": self.epsilon,
                    "total loss": avg_loss[0],
                    "dq loss": avg_loss[1],
                    "supervised loss": avg_loss[2],
                    "avg q values": avg_loss[3],
                    "demo num in minibatch": avg_loss[4],
                    "time per each step": avg_time_cost,
                }
            )

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
                log_value = (0, avg_loss, 0.0, t_end - t_begin)
                self.write_log(log_value)
        print("[INFO] Pre-Train Complete!\n")
