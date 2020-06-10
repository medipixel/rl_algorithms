# -*- coding: utf-8 -*-
"""PPO agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

import argparse
from typing import Tuple

import gym
import numpy as np
import torch
import wandb

from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.env.utils import env_generator, make_envs
from rl_algorithms.registry import AGENTS, build_learner
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class PPOAgent(Agent):
    """PPO Agent.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
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
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        learner_cfg: ConfigDict,
        log_cfg: ConfigDict,
    ):
        """Initialize.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        env_gen = env_generator(env.spec.id, args)
        env_multi = make_envs(env_gen, n_envs=hyper_params.n_workers)

        Agent.__init__(self, env, env_info, args, log_cfg)

        self.episode_steps = np.zeros(hyper_params.n_workers, dtype=np.int)
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.values: list = []
        self.masks: list = []
        self.log_probs: list = []
        self.i_episode = 0
        self.next_state = np.zeros((1,))

        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg
        self.learner_cfg.args = self.args
        self.learner_cfg.env_info = self.env_info
        self.learner_cfg.hyper_params = self.hyper_params
        self.learner_cfg.log_cfg = self.log_cfg
        self.learner_cfg.device = device

        if not self.args.test:
            self.env = env_multi

        self.epsilon = hyper_params.max_epsilon

        self.learner = build_learner(self.learner_cfg)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = torch.FloatTensor(state).to(device)
        selected_action, dist = self.learner.actor(state)

        if self.args.test and not self.is_discrete:
            selected_action = dist.mean

        if not self.args.test:
            value = self.learner.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool, dict]:
        next_state, reward, done, info = self.env.step(action.detach().cpu().numpy())

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = done.copy()
            done_bool[
                np.where(self.episode_steps == self.args.max_episode_steps)
            ] = False

            self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            self.masks.append(torch.FloatTensor(1 - done_bool).unsqueeze(1).to(device))

        return next_state, reward, done, info

    def decay_epsilon(self, t: int = 0):
        """Decay epsilon until reaching the minimum value."""
        max_epsilon = self.hyper_params.max_epsilon
        min_epsilon = self.hyper_params.min_epsilon
        epsilon_decay_period = self.hyper_params.epsilon_decay_period

        self.epsilon = self.epsilon - (max_epsilon - min_epsilon) * min(
            1.0, t / (epsilon_decay_period + 1e-7)
        )

    def write_log(
        self, log_value: tuple,
    ):
        i_episode, n_step, score, actor_loss, critic_loss, total_loss = log_value
        print(
            "[INFO] episode %d\tepisode steps: %d\ttotal score: %d\n"
            "total loss: %f\tActor loss: %f\tCritic loss: %f\n"
            % (i_episode, n_step, score, total_loss, actor_loss, critic_loss)
        )

        if self.args.log:
            wandb.log(
                {
                    "total loss": total_loss,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    "score": score,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.actor, self.critic], log="parameters")

        score = 0
        i_episode_prev = 0
        loss = [0.0, 0.0, 0.0]
        state = self.env.reset()

        while self.i_episode <= self.args.episode_num:
            for _ in range(self.hyper_params.rollout_len):
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.episode_steps += 1

                state = next_state
                score += reward[0]
                i_episode_prev = self.i_episode
                self.i_episode += done.sum()

                if (self.i_episode // self.args.save_period) != (
                    i_episode_prev // self.args.save_period
                ):
                    self.learner.save_params(self.i_episode)

                if done[0]:
                    n_step = self.episode_steps[0]
                    log_value = (
                        self.i_episode,
                        n_step,
                        score,
                        loss[0],
                        loss[1],
                        loss[2],
                    )
                    self.write_log(log_value)
                    score = 0

                self.episode_steps[np.where(done)] = 0
            self.next_state = next_state
            loss = self.learner.update_model(
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
