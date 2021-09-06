# -*- coding: utf-8 -*-
"""GAIL agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: eunjin.jung@medipixel.io
- Paper: https://arxiv.org/abs/1606.03476
"""

from typing import Tuple, Union

import gym
import numpy as np
import torch
import wandb

from rl_algorithms.common.buffer.gail_buffer import GAILBuffer
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.gail.utils import compute_gail_reward
from rl_algorithms.ppo.agent import PPOAgent
from rl_algorithms.registry import AGENTS
from rl_algorithms.utils.config import ConfigDict


@AGENTS.register_module
class GAILPPOAgent(PPOAgent):
    """PPO-based GAIL Agent.

    Attributes:
        env (gym.Env): openAI Gym environment
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values
        actor_optim (Optimizer): optimizer for training actor
        critic_optim (Optimizer): optimizer for training critic
        episode_steps (np.ndarray): step numbers of the current episode
        states (list): memory for experienced states
        actions (list): memory for experienced actions
        rewards (list): memory for experienced rewards
        values (list): memory for experienced values
        masks (list): memory for masks
        log_probs (list): memory for log_probs
        i_episode (int): current episode number
        epsilon (float): value for clipping loss

    """

    def __init__(
        self,
        env: gym.Env,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
        log_cfg: ConfigDict,
        is_test: bool,
        load_from: str,
        is_render: bool,
        render_after: int,
        is_log: bool,
        save_period: int,
        episode_num: int,
        max_episode_steps: int,
        interim_test_num: int,
    ):

        super().__init__(
            env,
            env_info,
            hyper_params,
            learner_cfg,
            log_cfg,
            is_test,
            load_from,
            is_render,
            render_after,
            is_log,
            save_period,
            episode_num,
            max_episode_steps,
            interim_test_num,
        )

        # load demo replay memory
        self.demo_memory = GAILBuffer(self.hyper_params.demo_path)
        self.learner.set_demo_memory(self.demo_memory)

    def step(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        next_state, reward, done, info = self.env.step(action)

        return next_state, reward, done, info

    def write_log(
        self,
        log_value: tuple,
    ):
        (
            i_episode,
            n_step,
            score,
            gail_reward,
            actor_loss,
            critic_loss,
            total_loss,
            discriminator_loss,
            discriminator_exp_acc,
            discriminator_demo_acc,
        ) = log_value
        print(
            "[INFO] episode %d\tepisode steps: %d\ttask score: %f\t gail score: %f\n"
            "total loss: %f\tActor loss: %f\tCritic loss: %f\t Discriminator loss: %f\n"
            "discriminator_exp_acc: %f\t discriminator_demo_acc: %f"
            % (
                i_episode,
                n_step,
                score,
                gail_reward,
                total_loss,
                actor_loss,
                critic_loss,
                discriminator_loss,
                discriminator_exp_acc,
                discriminator_demo_acc,
            )
        )

        if self.is_log:
            wandb.log(
                {
                    "total loss": total_loss,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    "discriminator_loss": discriminator_loss,
                    "gail reward": gail_reward,
                    "score": score,
                    "discriminator_exp_acc": discriminator_exp_acc,
                    "discriminator_demo_acc": discriminator_demo_acc,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.is_log:
            self.set_wandb()
            # wandb.watch([self.actor, self.critic], log="parameters")

        score = 0
        i_episode_prev = 0
        loss = [0.0, 0.0, 0.0, 0.0]
        discriminator_acc = [0.0, 0.0]
        state = self.env.reset()

        while self.i_episode <= self.episode_num:
            for _ in range(self.hyper_params.rollout_len):
                if self.is_render and self.i_episode >= self.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, task_reward, done, _ = self.step(action)

                # gail reward (imitation reward)
                gail_reward = compute_gail_reward(
                    self.learner.discriminator(
                        (
                            numpy2floattensor(state, self.learner.device),
                            numpy2floattensor(action, self.learner.device),
                        )
                    )
                )

                # hybrid reward
                # Reference: https://arxiv.org/abs/1802.09564
                reward = (
                    self.hyper_params.gail_reward_weight * gail_reward
                    + (1.0 - self.hyper_params.gail_reward_weight) * task_reward
                )

                if not self.is_test:
                    # if the last state is not a terminal state, store done as false
                    done_bool = done.copy()
                    done_bool[
                        np.where(self.episode_steps == self.max_episode_steps)
                    ] = False

                    self.rewards.append(
                        numpy2floattensor(reward, self.learner.device).unsqueeze(1)
                    )
                    self.masks.append(
                        numpy2floattensor(
                            (1 - done_bool), self.learner.device
                        ).unsqueeze(1)
                    )

                self.episode_steps += 1

                state = next_state
                score += task_reward[0]
                i_episode_prev = self.i_episode
                self.i_episode += done.sum()

                if (self.i_episode // self.save_period) != (
                    i_episode_prev // self.save_period
                ):
                    self.learner.save_params(self.i_episode)

                if done[0]:
                    n_step = self.episode_steps[0]
                    log_value = (
                        self.i_episode,
                        n_step,
                        score,
                        gail_reward,
                        loss[0],
                        loss[1],
                        loss[2],
                        loss[3],
                        discriminator_acc[0],
                        discriminator_acc[1],
                    )
                    self.write_log(log_value)
                    score = 0

                self.episode_steps[np.where(done)] = 0
            self.next_state = next_state
            loss, discriminator_acc = self.learner.update_model(
                (
                    self.states,
                    self.actions,
                    self.rewards,
                    self.values,
                    self.log_probs,
                    self.next_state,
                    self.masks,
                ),
                self.epsilon,
            )
            self.states, self.actions, self.rewards = [], [], []
            self.values, self.masks, self.log_probs = [], [], []
            self.decay_epsilon(self.i_episode)

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
