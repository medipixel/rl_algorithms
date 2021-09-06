# -*- coding: utf-8 -*-
"""DQfD agent using demo agent for episodic tasks in OpenAI Gym.

- Author: Kyunghwan Kim, Curt Park
- Contact: kh.kim@medipixel.io, curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1704.03732.pdf (DQfD)
"""

import pickle
import time

import numpy as np
import wandb

from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.registry import AGENTS, build_learner


@AGENTS.register_module
class DQfDAgent(DQNAgent):
    """
    DQN interacting with environment.

    Attribute:
        memory (PrioritizedReplayBuffer): replay memory

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.is_test:
            # load demo replay memory
            demos = self._load_demos()

            if self.use_n_step:
                demos, demos_n_step = common_utils.get_n_step_info_from_demo(
                    demos, self.hyper_params.n_step, self.hyper_params.gamma
                )

                self.memory_n = ReplayBuffer(
                    max_len=self.hyper_params.buffer_size,
                    batch_size=self.hyper_params.batch_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                    demo=demos_n_step,
                )

            # replay memory
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                demo=demos,
            )
            self.memory = PrioritizedBufferWrapper(
                self.memory,
                alpha=self.hyper_params.per_alpha,
                epsilon_d=self.hyper_params.per_eps_demo,
            )

        build_args = dict(
            hyper_params=self.hyper_params,
            log_cfg=self.log_cfg,
            env_name=self.env_info.name,
            state_size=self.env_info.observation_space.shape,
            output_size=self.env_info.action_space.n,
            is_test=self.is_test,
            load_from=self.load_from,
        )
        self.learner_cfg.type = "DQfDLearner"
        self.learner = build_learner(self.learner_cfg, build_args)

    def _load_demos(self) -> list:
        """Load expert's demonstrations."""
        # load demo replay memory
        with open(self.hyper_params.demo_path, "rb") as f:
            demos = pickle.load(f)

        return demos

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

        if self.is_log:
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
            experience = self.sample_experience()
            info = self.learner.update_model(experience)
            loss = info[0:5]
            t_end = time.time()
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                log_value = (0, avg_loss, 0.0, t_end - t_begin)
                self.write_log(log_value)
        print("[INFO] Pre-Train Complete!\n")

    def train(self):
        """Train the agent."""
        # logger
        if self.is_log:
            self.set_wandb()
            # wandb.watch([self.dqn], log="parameters")

        # pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.episode_num + 1):
            state = self.env.reset()
            self.episode_step = 0
            losses = list()
            done = False
            score = 0

            t_begin = time.time()

            while not done:
                if self.is_render and self.i_episode >= self.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            experience = self.sample_experience()
                            info = self.learner.update_model(experience)
                            loss = info[0:5]
                            indices, new_priorities = info[5:7]
                            losses.append(loss)  # for logging
                            self.memory.update_priorities(indices, new_priorities)

                    # decrease epsilon
                    self.epsilon = max(
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon)
                        * self.hyper_params.epsilon_decay,
                        self.min_epsilon,
                    )

                    # increase priority beta
                    fraction = min(float(self.i_episode) / self.episode_num, 1.0)
                    self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

                state = next_state
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)

            if self.i_episode % self.save_period == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()
