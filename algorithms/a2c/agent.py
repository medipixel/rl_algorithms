# -*- coding: utf-8 -*-
"""1-Step Advantage Actor-Critic agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from algorithms.common.abstract.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CAgent(Agent):
    """1-Step Advantage Actor-Critic interacting with environment.

    Attributes:
        actor (nn.Module): policy model to select actions
        critic (nn.Module): critic model to evaluate states
        hyper_params (dict): hyper-parameters
        actor_optimizer (Optimizer): optimizer for actor
        critic_optimizer (Optimizer): optimizer for critic
        optimizer (Optimizer): optimizer for training
        episode_step (int): step number of the current episode
        transition (list): recent transition information
        i_episode (int): current episode number

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic

        """
        Agent.__init__(self, env, args)

        self.actor, self.critic = models
        self.actor_optimizer, self.critic_optimizer = optims
        self.hyper_params = hyper_params
        self.log_prob = torch.zeros((1,))
        self.predicted_value = torch.zeros((1,))
        self.transition: list = list()
        self.episode_step = 0
        self.i_episode = 0

        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        state = torch.FloatTensor(state).to(device)

        selected_action, dist = self.actor(state)

        if self.args.test:
            selected_action = dist.mean
        else:
            predicted_value = self.critic(state)
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = []
            self.transition.extend([log_prob, predicted_value])

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        self.episode_step += 1

        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        if not self.args.test:
            done_bool = done
            if self.episode_step == self.args.max_episode_steps:
                done_bool = False
            self.transition.extend([next_state, reward, done_bool])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        log_prob, pred_value, next_state, reward, done = self.transition
        next_state = torch.FloatTensor(next_state).to(device)

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        next_value = self.critic(next_state).detach()
        q_value = reward + self.hyper_params["GAMMA"] * next_value * mask
        q_value = q_value.to(device)

        # advantage = Q_t - V(s_t)
        advantage = q_value - pred_value

        # calculate loss at the current step
        policy_loss = -advantage.detach() * log_prob  # adv. is not backpropagated
        policy_loss += self.hyper_params["W_ENTROPY"] * -log_prob  # entropy
        value_loss = F.smooth_l1_loss(pred_value, q_value.detach())

        # train
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.hyper_params["GRADIENT_CLIP_AC"]
        )
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.hyper_params["GRADIENT_CLIP_CR"]
        )
        self.critic_optimizer.step()

        return policy_loss.data, value_loss.data

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[INFO] The input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] Loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }

        Agent.save_params(self, params, n_episode)

    def write_log(self, i: int, score: int, policy_loss: float, value_loss: float):
        total_loss = policy_loss + value_loss

        print(
            "[INFO] episode %d\tepisode step: %d\ttotal score: %d\n"
            "total loss: %.4f\tpolicy loss: %.4f\tvalue loss: %.4f\n"
            % (i, self.episode_step, score, total_loss, policy_loss, value_loss)
        )

        if self.args.log:
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
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            # wandb.watch([self.actor, self.critic], log="parameters")

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            policy_loss_episode = list()
            value_loss_episode = list()
            self.episode_step = 0

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                policy_loss, value_loss = self.update_model()

                policy_loss_episode.append(policy_loss)
                value_loss_episode.append(value_loss)

                state = next_state
                score += reward

            # logging
            policy_loss = np.array(policy_loss_episode).mean()
            value_loss = np.array(value_loss_episode).mean()
            self.write_log(self.i_episode, score, policy_loss, value_loss)

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.save_params(self.i_episode)
        self.interim_test()
