# -*- coding: utf-8 -*-
"""DQfD2 for episodic tasks in OpenAI Gym.

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
        dq_loss = torch.mean((target - curr_q_values).pow(2))
        dq_loss *= self.lambda1

        # get margin for each demo transitions
        q_values_demo = self.dqn(demo_states, self.epsilon)
        action_idxs = demo_actions.long()
        margin = torch.ones(q_values_demo.size()) * self.hyper_params["MARGIN"]
        margin[:, action_idxs] = 0.0  # demo actions have 0 margins
        margin = margin.to(device)

        # calculate supervised loss
        q_value_demo = q_values_demo[:, action_idxs].squeeze()
        max_margin_q_values = torch.max(q_values_demo + margin, dim=-1)[0]
        supervised_loss = torch.mean(max_margin_q_values - q_value_demo)
        supervised_loss *= self.lambda2

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params["W_Q_REG"]

        # total loss
        loss = dq_loss + supervised_loss + q_regular

        # train dqn
        self.dqn_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params["GRADIENT_CLIP"])
        self.dqn_optimizer.step()

        # update target networks
        tau = self.hyper_params["TAU"]
        common_utils.soft_update(self.dqn, self.dqn_target, tau)

        return loss.data, dq_loss.data, supervised_loss.data

    def write_log(self, i: int, avg_loss: np.ndarray, score: int = 0):
        """Write log about loss and score"""
        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "epsilon: %f, total loss: %f, dq loss: %f, supervised_loss loss: %f\n"
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
