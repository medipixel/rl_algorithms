# -*- coding: utf-8 -*-
"""DQfD agent using demo agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim, Curt Park
- Contact: kh.kim@medipixel.io, curt.park@medipixel.io
- Paper: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf (DQN)
         https://arxiv.org/pdf/1509.06461.pdf (Double DQN)
         https://arxiv.org/pdf/1511.05952.pdf (PER)
         https://arxiv.org/pdf/1511.06581.pdf (Dueling)
         https://arxiv.org/pdf/1704.03732.pdf (DQfD)
"""

import datetime
import pickle
from typing import Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import wandb

from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBufferfD
import algorithms.common.helper_functions as common_utils
from algorithms.dqn.agent import Agent as DQNAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(DQNAgent):
    """DQN interacting with environment.

    Attribute:
        memory (PrioritizedReplayBufferfD): replay memory

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

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones, weights, indexes, eps_d = (
            experiences
        )

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
        dq_loss_element_wise = (target - curr_q_values).pow(2)
        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # supervised loss using demo for only demo transitions
        demo_idxs = np.where(eps_d != 0.0)
        if demo_idxs[0].size != 0:  # if 1 or more demos are sampled
            # get margin for each demo transition
            action_idxs = actions[demo_idxs].long()
            margin = torch.ones(q_values.size()) * self.hyper_params["MARGIN"]
            margin[demo_idxs, action_idxs] = 0.0  # demo actions have 0 margins
            margin = margin.to(device)

            # calculate supervised loss
            demo_q_values = q_values[demo_idxs, action_idxs].squeeze()
            supervised_loss = torch.max(q_values + margin, dim=-1)[0]
            supervised_loss = supervised_loss[demo_idxs] - demo_q_values
            supervised_loss = torch.mean(supervised_loss) * self.hyper_params["LAMBDA2"]
        else:  # no demo sampled
            supervised_loss = torch.zeros(1).to(device)

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

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy().squeeze()
        new_priorities = loss_for_prior + self.hyper_params["PER_EPS"]
        new_priorities += eps_d
        self.memory.update_priorities(indexes, new_priorities)

        # increase beta
        fraction = min(float(self.i_episode) / self.args.max_episode_steps, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        return loss.data, dq_loss.data, supervised_loss.data

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
