# -*- coding: utf-8 -*-
"""DQN with Behaviour Cloning agent (w/o HER) for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
         https://arxiv.org/pdf/1509.06461.pdf (Double DQN)
         https://arxiv.org/pdf/1709.10089.pdf (Behaviour Cloning)
"""

import datetime
import pickle
from typing import Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import wandb

from algorithms.common.buffer.replay_buffer import ReplayBuffer
import algorithms.common.helper_functions as common_utils
from algorithms.dqn.agent import Agent as DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(DQNAgent):
    """DQN interacting with environment.

    Attribute:
        memory (ReplayBuffer): replay memory
        demo_memory (ReplayBuffer): replay memory for demo

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        # load demo replay memory
        if not self.args.test:
            with open(self.args.demo_path, "rb") as f:
                demo = list(pickle.load(f))

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

        q_values = self.dqn(states, self.epsilon)
        next_q_values = self.dqn(next_states, self.epsilon)
        next_target_q_values = self.dqn_target(next_states, self.epsilon)

        curr_q_values = q_values.gather(1, actions.long().unsqueeze(1))
        next_q_values = next_target_q_values.gather(  # Double DQN
            1, next_q_values.argmax(1).unsqueeze(1)
        )

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        target = rewards + self.hyper_params["GAMMA"] * next_q_values * masks
        target = target.to(device)

        # calculate dq loss
        dq_loss = (target - curr_q_values).pow(2)

        # calculate bc loss
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

        bc_loss *= self.lambda2

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params["W_Q_REG"]

        # total loss
        loss = dq_loss + bc_loss + q_regular

        # train dqn
        self.dqn_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params["GRADIENT_CLIP"])
        self.dqn_optimizer.step()

        # update target networks
        tau = self.hyper_params["TAU"]
        common_utils.soft_update(self.dqn, self.dqn_target, tau)

        return loss.data, dq_loss.data

    def write_log(self, i: int, avg_loss: np.ndarray, score: int = 0):
        """Write log about loss and score"""
        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "epsilon: %f, total loss: %f, dq loss: %f, supervised loss: %f\n"
            "at %s\n"
            % (
                i,
                self.episode_steps[0],
                self.total_steps.sum(),
                score,
                self.epsilon,
                avg_loss[0],
                avg_loss[1],
                avg_loss[2],
                datetime.datetime.now(),
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
                }
            )

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
                self.write_log(0, avg_loss)
