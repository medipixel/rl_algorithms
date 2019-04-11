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
import wandb

from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.env.multiprocessing_env import SubprocVecEnv
import algorithms.ppo.utils as ppo_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """PPO Agent.

    Attributes:
        env (gym.Env or SubprocVecEnv): Gym env with multiprocessing for training
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        hyper_params (dict): hyper-parameters
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
        env_single: gym.Env,  # for testing
        env_multi: SubprocVecEnv,  # for training
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
    ):
        """Initialization.

        Args:
            env_single (gym.Env): openAI Gym environment for testing
            env_multi (SubprocVecEnv): Gym env with multiprocessing for training
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic

        """
        AbstractAgent.__init__(self, env_single, args)

        if not self.args.test:
            self.env = env_multi
        self.actor, self.critic = models
        self.actor_optimizer, self.critic_optimizer = optims
        self.epsilon = hyper_params["EPSILON"]
        self.hyper_params = hyper_params
        self.episode_steps = np.zeros(hyper_params["N_WORKERS"], dtype=np.int)
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.values: list = []
        self.masks: list = []
        self.log_probs: list = []
        self.i_episode = 0

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

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        self.episode_steps += 1

        next_state, reward, done, _ = self.env.step(action.detach().cpu().numpy())

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = done.copy()
            done_bool[
                np.where(self.episode_steps == self.args.max_episode_steps)
            ] = False

            self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            self.masks.append(torch.FloatTensor(1 - done_bool).unsqueeze(1).to(device))

        return next_state, reward, done

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

        if self.hyper_params["STANDARDIZE_ADVANTAGE"]:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        actor_losses, critic_losses, total_losses = [], [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_utils.ppo_iter(
            self.hyper_params["EPOCH"],
            self.hyper_params["BATCH_SIZE"],
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
            if self.hyper_params["USE_CLIPPED_VALUE_LOSS"]:
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
                actor_loss
                + self.hyper_params["W_VALUE"] * critic_loss
                - self.hyper_params["W_ENTROPY"] * entropy
            )

            # train critic
            self.critic_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.hyper_params["GRADIENT_CLIP_AC"]
            )
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.hyper_params["GRADIENT_CLIP_CR"]
            )
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
        max_epsilon = self.hyper_params["EPSILON"]
        min_epsilon = self.hyper_params["MIN_EPSILON"]
        epsilon_decay_period = self.hyper_params["EPSILON_DECAY_PERIOD"]

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
        AbstractAgent.save_params(self, params, n_episode)

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
            wandb.init()
            wandb.config.update(self.hyper_params)
            # wandb.watch([self.actor, self.critic], log="parameters")

        score = 0
        i_episode_prev = 0
        loss = [0.0, 0.0, 0.0]
        state = self.env.reset()

        while self.i_episode <= self.args.episode_num:
            for _ in range(self.hyper_params["ROLLOUT_LEN"]):
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

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
