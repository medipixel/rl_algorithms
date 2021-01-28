# -*- coding: utf-8 -*-
"""1-Step Advantage Actor-Critic agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Tuple

import gym
import numpy as np
import torch
import wandb

from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.registry import AGENTS, build_learner
from rl_algorithms.utils.config import ConfigDict


@AGENTS.register_module
class A2CAgent(Agent):
    """1-Step Advantage Actor-Critic interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        actor (nn.Module): policy model to select actions
        critic (nn.Module): critic model to evaluate states
        actor_optim (Optimizer): optimizer for actor
        critic_optim (Optimizer): optimizer for critic
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        transition (list): recent transition information

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
        """Initialize."""
        Agent.__init__(
            self,
            env,
            env_info,
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

        self.transition: list = list()
        self.episode_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg

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

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = numpy2floattensor(state, self.learner.device)

        selected_action, dist = self.learner.actor(state)

        if self.is_test:
            selected_action = dist.mean
        else:
            predicted_value = self.learner.critic(state)
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = []
            self.transition.extend([log_prob, predicted_value])

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""

        action = action.detach().cpu().numpy()
        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            done_bool = done
            if self.episode_step == self.max_episode_steps:
                done_bool = False
            self.transition.extend([next_state, reward, done_bool])

        return next_state, reward, done, info

    def write_log(self, log_value: tuple):
        i, score, policy_loss, value_loss = log_value
        total_loss = policy_loss + value_loss

        print(
            "[INFO] episode %d\tepisode step: %d\ttotal score: %d\n"
            "total loss: %.4f\tpolicy loss: %.4f\tvalue loss: %.4f\n"
            % (i, self.episode_step, score, total_loss, policy_loss, value_loss)
        )

        if self.is_log:
            wandb.log(
                {
                    "total loss": total_loss,
                    "policy loss": policy_loss,
                    "value loss": value_loss,
                    "score": score,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.is_log:
            self.set_wandb()
            # wandb.watch([self.actor, self.critic], log="parameters")

        for self.i_episode in range(1, self.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            policy_loss_episode = list()
            value_loss_episode = list()
            self.episode_step = 0

            while not done:
                if self.is_render and self.i_episode >= self.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.episode_step += 1

                policy_loss, value_loss = self.learner.update_model(self.transition)

                policy_loss_episode.append(policy_loss)
                value_loss_episode.append(value_loss)

                state = next_state
                score += reward

            # logging
            policy_loss = np.array(policy_loss_episode).mean()
            value_loss = np.array(value_loss_episode).mean()
            log_value = (self.i_episode, score, policy_loss, value_loss)
            self.write_log(log_value)

            if self.i_episode % self.save_period == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()
