# -*- coding: utf-8 -*-
"""PPO agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from algorithms.common.abstract.agent import Agent
from algorithms.common.env.multiprocessing_env import SubprocVecEnv
from algorithms.common.env.utils import env_generator, make_envs
from algorithms.common.networks.mlp import MLP, GaussianDist
import algorithms.ppo.utils as ppo_utils
from algorithms.registry import AGENTS
from algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class PPOAgent(Agent):
    """PPO Agent.

    Attributes:
        env (gym.Env or SubprocVecEnv): Gym env with multiprocessing for training
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        episode_steps (np.ndarray): step numbers of the current episode
        states (list): memory for experienced states
        actions (list): memory for experienced actions
        rewards (list): memory for experienced rewards
        values (list): memory for experienced values
        masks (list): memory for masks
        log_probs (list): memory for log_probs
        i_episode (int): current episode number

    """

    def __init__(
        self,
        env: gym.Env,  # for testing
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        gamma: float,
        batch_size: int,
        initial_random_action: int,
        lambda_: int,
        epsilon: float,
        min_epsilon: float,
        epsilon_decay_period: int,
        w_value: float,
        w_entropy: float,
        gradient_clip_ac: float,
        gradient_clip_cr: float,
        epoch: int,
        rollout_len: int,
        n_workers: int,
        use_clipped_value_loss: bool,
        standardize_advantage: bool,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        env_single = env
        env_gen = env_generator(env.spec.id, args)
        env_multi = make_envs(env_gen, n_envs=n_workers)

        Agent.__init__(self, env_single, args, log_cfg)

        self.episode_steps = np.zeros(n_workers, dtype=np.int)
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.values: list = []
        self.masks: list = []
        self.log_probs: list = []
        self.i_episode = 0

        self.gamma = gamma
        self.batch_size = batch_size
        self.initial_random_action = initial_random_action
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_period = epsilon_decay_period
        self.w_value = w_value
        self.w_entropy = w_entropy
        self.gradient_clip_ac = gradient_clip_ac
        self.gradient_clip_cr = gradient_clip_cr
        self.epoch = epoch
        self.rollout_len = rollout_len
        self.use_clipped_value_loss = use_clipped_value_loss
        self.standardize_advantage = standardize_advantage

        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        if not self.args.test:
            self.env = env_multi

        self.actor = GaussianDist(
            input_size=state_dim,
            output_size=action_dim,
            hidden_sizes=network_cfg.hidden_sizes_actor,
            hidden_activation=torch.tanh,
        ).to(device)

        self.critic = MLP(
            input_size=state_dim,
            output_size=1,
            hidden_sizes=network_cfg.hidden_sizes_critic,
            hidden_activation=torch.tanh,
        ).to(device)

        # create optimizer
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=optim_cfg.lr_actor,
            weight_decay=optim_cfg.weight_decay,
        )

        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=optim_cfg.lr_critic,
            weight_decay=optim_cfg.weight_decay,
        )

        # load model parameters
        if self.args.load_from is not None and os.path.exists(self.args.load_from):
            self.load_params(self.args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = torch.FloatTensor(state).to(device)
        selected_action, dist = self.actor(state)

        if self.args.test and not self.is_discrete:
            selected_action = dist.mean

        if not self.args.test:
            value = self.critic(state)
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

    def update_model(
        self, next_state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the model after every N episodes."""

        next_state = torch.FloatTensor(next_state).to(device)
        next_value = self.critic(next_state)

        returns = ppo_utils.compute_gae(
            next_value, self.rewards, self.masks, self.values
        )

        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        if self.is_discrete:
            actions = actions.unsqueeze(1)
            log_probs = log_probs.unsqueeze(1)

        if self.starndardize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        actor_losses, critic_losses, total_losses = [], [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_utils.ppo_iter(
            self.epoch,
            self.batch_size,
            states,
            actions,
            values,
            log_probs,
            returns,
            advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            surr_loss = ratio * adv
            clipped_surr_loss = (
                torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            )
            actor_loss = -torch.min(surr_loss, clipped_surr_loss).mean()

            # critic_loss
            value = self.critic(state)
            if self.use_clipped_value_loss:
                value_pred_clipped = old_value + torch.clamp(
                    (value - old_value), -self.epsilon, self.epsilon
                )
                value_loss_clipped = (return_ - value_pred_clipped).pow(2)
                value_loss = (return_ - value).pow(2)
                critic_loss = 0.5 * torch.max(value_loss, value_loss_clipped).mean()
            else:
                critic_loss = 0.5 * (return_ - value).pow(2).mean()

            # entropy
            entropy = dist.entropy().mean()

            # total_loss
            total_loss = (
                actor_loss + self.w_value * critic_loss - self.w_entropy * entropy
            )

            # train critic
            self.critic_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_ac)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_cr)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            total_losses.append(total_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        total_loss = sum(total_losses) / len(total_losses)

        return actor_loss, critic_loss, total_loss

    def decay_epsilon(self, t: int = 0):
        """Decay epsilon until reaching the minimum value."""
        max_epsilon = self.epsilon
        min_epsilon = self.min_epsilon
        epsilon_decay_period = self.epsilon_decay_period

        self.epsilon = max_epsilon - (max_epsilon - min_epsilon) * min(
            1.0, t / (epsilon_decay_period + 1e-7)
        )

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }
        Agent.save_params(self, params, n_episode)

    def write_log(
        self,
        i_episode: int,
        n_step: int,
        score: int,
        actor_loss: float,
        critic_loss: float,
        total_loss: float,
    ):
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
            self.set_wandb(is_training=True)
            # wandb.watch([self.actor, self.critic], log="parameters")

        score = 0
        i_episode_prev = 0
        loss = [0.0, 0.0, 0.0]
        state = self.env.reset()

        while self.i_episode <= self.args.episode_num:
            for _ in range(self.rollout_len):
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
                    self.save_params(self.i_episode)

                if done[0]:
                    n_step = self.episode_steps[0]
                    self.write_log(
                        self.i_episode, n_step, score, loss[0], loss[1], loss[2]
                    )
                    score = 0

                self.episode_steps[np.where(done)] = 0

            loss = self.update_model(next_state)
            self.decay_epsilon(self.i_episode)

        # termination
        self.env.close()
        self.save_params(self.i_episode)
