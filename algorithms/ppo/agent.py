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

import algorithms.ppo.utils as ppo_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.env.multiprocessing_env import SubprocVecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """PPO Agent.

    Attributes:
        envs (SubprocVecEnv): Gym env with multiprocessing for training
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        hyper_params (dict): hyper-parameters
        episode_step (int): step number of the current episode
        states (list): memory for experienced states
        actions (list): memory for experienced actions
        rewards (list): memory for experienced rewards
        values (list): memory for experienced values
        masks (list): memory for masks
        log_probs (list): memory for log_probs

    """

    def __init__(
        self,
        env: gym.Env,  # for testing
        envs: SubprocVecEnv,  # for training
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment for testing
            envs (SubprocVecEnv): Gym env with multiprocessing for training
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic

        """
        AbstractAgent.__init__(self, env, args)

        self.envs = envs
        self.actor, self.critic = models
        self.actor_optimizer, self.critic_optimizer = optims
        self.epsilon = hyper_params["EPSILON"]
        self.hyper_params = hyper_params
        self.episode_step = 0
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.values: list = []
        self.masks: list = []
        self.log_probs: list = []

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
        self.episode_step += 1

        next_state, reward, done, _ = self.envs.step(action.detach().cpu().numpy())

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
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
                self.critic.parameters(), self.hyper_params["GRADIENT_CLIP"]
            )
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.hyper_params["GRADIENT_CLIP"]
            )
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.data)
            critic_losses.append(critic_loss.data)
            total_losses.append(total_loss.data)

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
        self, i_episode: int, actor_loss: float, critic_loss: float, total_loss: float
    ):
        n_test = self.hyper_params["N_TEST"]
        avg_step, avg_score = 0.0, 0.0

        for _ in range(n_test):
            step, score = self.run_test_env(
                self.args.render and i_episode >= self.args.render_after
            )
            avg_step += step / n_test
            avg_score += score / n_test

        print(
            "[INFO] episode %d\tepisode steps: %d\ttotal score: %d\n"
            "total loss: %f\tActor loss: %f\tCritic loss: %f\n"
            % (i_episode, avg_step, avg_score, total_loss, actor_loss, critic_loss)
        )

        if self.args.log:
            wandb.log(
                {
                    "total loss": total_loss,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    "score": avg_score,
                }
            )

    def run_test_env(self, render: bool) -> Tuple[int, int]:
        """Run the agent on the test env for evaluation."""
        state = self.env.reset()

        done = False
        score = 0
        step = 0
        while not done:
            if render:
                self.env.render()

            action, dist = self.actor(torch.FloatTensor(state).to(device))

            if not self.is_discrete:
                action = dist.mean

            next_state, reward, done, _ = self.env.step(action.detach().cpu().numpy())

            state = next_state
            score += reward
            step += 1

        return step, score

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            wandb.watch([self.actor, self.critic], log="parameters")

        i_episode = 1
        print_log = False
        state = self.envs.reset()

        while i_episode <= self.args.episode_num:
            for _ in range(self.hyper_params["ROLLOUT_LEN"]):
                action = self.select_action(state)
                next_state, _, done = self.step(action)

                state = next_state
                if done[0]:
                    i_episode += 1
                    print_log = True
                    self.episode_step = 0

                    if i_episode % self.args.save_period == 0:
                        self.save_params(i_episode)

            loss = self.update_model(next_state)
            self.decay_epsilon(i_episode)

            if print_log:
                self.write_log(i_episode, loss[0], loss[1], loss[2])
                print_log = False

        # termination
        self.envs.close()
        self.env.close()
        self.save_params(i_episode)

    def test(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()

        for i_episode in range(1, self.args.episode_num + 1):
            step, score = self.run_test_env(
                self.args.render and i_episode >= self.args.render_after
            )

            print(
                "[INFO] episode %d\tstep: %d\ttotal score: %d"
                % (i_episode, step, score)
            )

            if self.args.log:
                wandb.log({"score": score})
