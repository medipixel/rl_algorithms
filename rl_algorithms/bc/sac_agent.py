# -*- coding: utf-8 -*-
"""BC with SAC agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
         https://arxiv.org/pdf/1709.10089.pdf
"""

import pickle
import time
from typing import Tuple

import numpy as np
import torch
import wandb

from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.registry import AGENTS, build_her, build_learner
from rl_algorithms.sac.agent import SACAgent


@AGENTS.register_module
class BCSACAgent(SACAgent):
    """BC with SAC agent interacting with environment.

    Attrtibutes:
        her (HER): hinsight experience replay
        transitions_epi (list): transitions per episode (for HER)
        desired_state (np.ndarray): desired state of current episode
        memory (ReplayBuffer): replay memory
        demo_memory (ReplayBuffer): replay memory for demo
        lambda2 (float): proportion of BC loss

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        # load demo replay memory
        with open(self.hyper_params.demo_path, "rb") as f:
            demo = list(pickle.load(f))

        # HER
        if self.hyper_params.use_her:
            self.her = build_her(self.hyper_params.her)
            print(f"[INFO] Build {str(self.her)}.")

            if self.hyper_params.desired_states_from_demo:
                self.her.fetch_desired_states_from_demo(demo)

            self.transitions_epi: list = list()
            self.desired_state = np.zeros((1,))
            demo = self.her.generate_demo_transitions(demo)

            if not self.her.is_goal_in_state:
                self.state_dim = (self.state_dim[0] * 2,)
        else:
            self.her = None

        if not self.is_test:
            # Replay buffers
            demo_batch_size = self.hyper_params.demo_batch_size
            self.demo_memory = ReplayBuffer(len(demo), demo_batch_size)
            self.demo_memory.extend(demo)

            self.memory = ReplayBuffer(self.hyper_params.buffer_size, demo_batch_size)

            # set hyper parameters
            self.hyper_params["lambda2"] = 1.0 / demo_batch_size

        build_args = dict(
            hyper_params=self.hyper_params,
            log_cfg=self.log_cfg,
            env_name=self.env_info.name,
            state_size=self.env_info.observation_space.shape,
            output_size=self.env_info.action_space.shape[0],
            is_test=self.is_test,
            load_from=self.load_from,
        )
        self.learner = build_learner(self.learner_cfg, build_args)

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        if self.hyper_params.use_her:
            self.desired_state = self.her.get_desired_state()
            state = np.concatenate((state, self.desired_state), axis=-1)
        state = numpy2floattensor(state, self.learner.device)
        return state

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        if self.hyper_params.use_her:
            self.transitions_epi.append(transition)
            done = transition[-1] or self.episode_step == self.max_episode_steps
            if done:
                # insert generated transitions if the episode is done
                transitions = self.her.generate_transitions(
                    self.transitions_epi,
                    self.desired_state,
                    self.hyper_params.success_score,
                )
                self.memory.extend(transitions)
                self.transitions_epi.clear()
        else:
            self.memory.add(transition)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, policy_update_freq, avg_time_cost = log_value
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "total loss: %.3f actor_loss: %.3f qf_1_loss: %.3f qf_2_loss: %.3f "
            "vf_loss: %.3f alpha_loss: %.3f n_qf_mask: %d (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0] * policy_update_freq,  # actor loss
                loss[1],  # qf_1 loss
                loss[2],  # qf_2 loss
                loss[3],  # vf loss
                loss[4],  # alpha loss
                loss[5],  # n_qf_mask
                avg_time_cost,
            )
        )

        if self.is_log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * policy_update_freq,
                    "qf_1 loss": loss[1],
                    "qf_2 loss": loss[2],
                    "vf loss": loss[3],
                    "alpha loss": loss[4],
                    "time per each step": avg_time_cost,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.is_log:
            self.set_wandb()
            # wandb.watch([self.actor, self.vf, self.qf_1, self.qf_2], log="parameters")

        # pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            loss_episode = list()

            t_begin = time.time()

            while not done:
                if self.is_render and self.i_episode >= self.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                state = next_state
                score += reward

                # training
                if len(self.memory) >= self.hyper_params.batch_size:
                    for _ in range(self.hyper_params.multiple_update):
                        experience = self.memory.sample()
                        demos = self.demo_memory.sample()
                        experience, demo = (
                            numpy2floattensor(experience, self.learner.device),
                            numpy2floattensor(demos, self.learner.device),
                        )
                        loss = self.learner.update_model(experience, demo)
                        loss_episode.append(loss)  # for logging

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                log_value = (
                    self.i_episode,
                    avg_loss,
                    score,
                    self.hyper_params.policy_update_freq,
                    avg_time_cost,
                )
                self.write_log(log_value)

            if self.i_episode % self.save_period == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()
