# -*- coding: utf-8 -*-
"""TD3 agent with PER for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1802.09477.pdf
         https://arxiv.org/pdf/1511.05952.pdf
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.optim as optim
import wandb

import algorithms.common.utils.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.noise.gaussian_noise import GaussianNoise
from algorithms.common.replaybuffer.priortized_replay_buffer import (
    PrioritizedReplayBuffer,
)
from algorithms.td3.model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 1e-3,
    "NOISE_STD": 1.0,
    "NOISE_CLIP": 0.5,
    "DELAYED_UPDATE": 2,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 128,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC_1": 1e-3,
    "LR_CRITIC_2": 1e-3,
    "GAUSSIAN_NOISE_MIN_SIGMA": 1.0,
    "GAUSSIAN_NOISE_MAX_SIGMA": 1.0,
    "GAUSSIAN_NOISE_DECAY_PERIOD": 1000000,
    "PER_ALPHA": 0.5,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
}


class Agent(AbstractAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (PrioritizedReplayBuffer): replay memory
        noise (GaussianNoise): random noise for exploration
        actor (nn.Module): actor model to select actions
        critic_1 (nn.Module): critic model to predict state values
        critic_2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        actor_target (nn.Module): target actor model to select actions
        critic_optimizer1 (Optimizer): optimizer for training critic_1
        critic_optimizer2 (Optimizer): optimizer for training critic_2
        actor_optimizer (Optimizer): optimizer for training actor
        curr_state (np.ndarray): temporary storage of the current state
        n_step (int): iteration number of the current episode

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment with discrete action space
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        AbstractAgent.__init__(self, env, args)

        self.curr_state = np.zeros((self.state_dim,))
        self.n_step = 0

        # create actor
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # create critic
        self.critic_1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_2 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target2 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target1.load_state_dict(self.critic_1.state_dict())
        self.critic_target2.load_state_dict(self.critic_2.state_dict())

        # create optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=hyper_params["LR_ACTOR"]
        )
        self.critic_optimizer1 = optim.Adam(
            self.critic_1.parameters(), lr=hyper_params["LR_CRITIC_1"]
        )
        self.critic_optimizer2 = optim.Adam(
            self.critic_2.parameters(), lr=hyper_params["LR_CRITIC_2"]
        )

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # noise instance to make randomness of action
        self.noise = GaussianNoise(
            self.args.seed,
            hyper_params["GAUSSIAN_NOISE_MIN_SIGMA"],
            hyper_params["GAUSSIAN_NOISE_MAX_SIGMA"],
            hyper_params["GAUSSIAN_NOISE_DECAY_PERIOD"],
        )

        # replay memory
        self.beta = hyper_params["PER_BETA"]
        self.memory = PrioritizedReplayBuffer(
            hyper_params["BUFFER_SIZE"],
            hyper_params["BATCH_SIZE"],
            self.args.seed,
            alpha=hyper_params["PER_ALPHA"],
        )

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        self.curr_state = state

        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state)

        action_size = selected_action.size()
        selected_action += torch.FloatTensor(
            self.noise.sample(action_size, self.n_step)
        ).to(device)

        return torch.clamp(selected_action, -1.0, 1.0)

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        self.memory.add(self.curr_state, action, reward, next_state, done)

        return next_state, reward, done

    def update_model(
        self, experiences: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones, weights, indexes = experiences
        masks = 1 - dones

        # get actions with noise
        noise_std, noise_clip = hyper_params["NOISE_STD"], hyper_params["NOISE_CLIP"]
        next_actions = self.actor_target(next_states)
        noise = torch.normal(torch.zeros(next_actions.size()), noise_std).to(device)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_actions += noise

        # min (Q_1', Q_2')
        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + hyper_params["GAMMA"] * next_values * masks
        curr_returns = curr_returns.to(device).detach()

        # critic loss
        values1 = self.critic_1(states, actions)
        values2 = self.critic_2(states, actions)
        critic_loss1 = torch.mean((values1 - curr_returns).pow(2) * weights)
        critic_loss2 = torch.mean((values2 - curr_returns).pow(2) * weights)

        # train critic
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if self.n_step % hyper_params["DELAYED_UPDATE"] == 0:
            # train actor
            actions = self.actor(states)
            actor_loss = torch.mean(-self.critic_1(states, actions) * weights)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            tau = hyper_params["TAU"]
            common_utils.soft_update(self.critic_1, self.critic_target1, tau)
            common_utils.soft_update(self.critic_2, self.critic_target2, tau)
            common_utils.soft_update(self.actor, self.actor_target, tau)
        else:
            actor_loss = torch.zeros(1)

        # update priorities in PER
        new_priorities = (torch.min(values1, values2) - curr_returns).pow(2)
        new_priorities = new_priorities.data.cpu().numpy() + hyper_params["PER_EPS"]
        self.memory.update_priorities(indexes, new_priorities)

        return actor_loss.data, critic_loss1.data, critic_loss2.data

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.critic_1.load_state_dict(params["critic_1"])
        self.critic_2.load_state_dict(params["critic_2"])
        self.critic_target1.load_state_dict(params["critic_target1"])
        self.critic_target2.load_state_dict(params["critic_target2"])
        self.critic_optimizer1.load_state_dict(params["critic_optim1"])
        self.critic_optimizer2.load_state_dict(params["critic_optim2"])
        self.actor.load_state_dict(params["actor"])
        self.actor_target.load_state_dict(params["actor_target"])
        self.actor_optimizer.load_state_dict(params["actor_optim"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_target1": self.critic_target1.state_dict(),
            "critic_target2": self.critic_target2.state_dict(),
            "critic_optim1": self.critic_optimizer1.state_dict(),
            "critic_optim2": self.critic_optimizer2.state_dict(),
        }

        AbstractAgent.save_params(self, self.args.algo, params, n_episode)

    def write_log(
        self, i: int, loss: np.ndarray, score: float = 0.0, delayed_update: int = 1
    ):
        """Write log about loss and score"""
        total_loss = loss.sum()
        print(
            "[INFO] episode %d total score: %d, total loss: %f\n"
            "actor_loss: %.3f critic_1_loss: %.3f critic_2_loss: %.3f\n"
            % (
                i,
                score,
                total_loss,
                loss[0] * delayed_update,  # actor loss
                loss[1],  # critic1 loss
                loss[2],  # critic2 loss
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * delayed_update,
                    "critic_1 loss": loss[1],
                    "critic_2 loss": loss[2],
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor, self.critic_1, self.critic_2], log="parameters")

        step = 0
        for i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                if len(self.memory) >= hyper_params["BATCH_SIZE"]:
                    experiences = self.memory.sample(self.beta)
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging

                # increase beta
                fraction = min(float(i_episode) / self.args.max_episode_steps, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                state = next_state
                score += reward
                step += 1

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(
                    i_episode, avg_loss, score, hyper_params["DELAYED_UPDATE"]
                )

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
