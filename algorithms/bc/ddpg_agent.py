# -*- coding: utf-8 -*-
"""DDPG with Behavior Cloning agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1709.10089.pdf
"""


import argparse
import os
import pickle
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

import algorithms.common.utils.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.noise.ou_noise import OUNoise
from algorithms.common.replaybuffer.replay_buffer import ReplayBuffer
from algorithms.ddpg.model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.99,
    "TAU": 1e-3,
    "BUFFER_SIZE": int(1e5),
    "BATCH_SIZE": 1024,
    "DEMO_BATCH_SIZE": 128,
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    "OU_NOISE_THETA": 0.0,
    "OU_NOISE_SIGMA": 0.0,
    "LAMBDA1": 1e-3,
    "LAMBDA2": 1.0,
}


class Agent(AbstractAgent):
    """ActorCritic interacting with environment.

    Attributes:
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        noise (OUNoise): random noise for exploration
        memory (ReplayBuffer): replay memory
        curr_state (np.ndarray): temporary storage of the current state

    """

    def __init__(self, env: gym.Env, args: argparse.Namespace):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment with discrete action space
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        AbstractAgent.__init__(self, env, args)

        self.curr_state = np.zeros((self.state_dim,))

        # create actor
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # create critic
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=hyper_params["LR_ACTOR"]
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=hyper_params["LR_CRITIC"]
        )

        # set hyper parameters
        self.lambda1 = hyper_params["LAMBDA1"]
        self.lambda2 = hyper_params["LAMBDA2"] / hyper_params["DEMO_BATCH_SIZE"]

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # noise instance to make randomness of action
        self.noise = OUNoise(
            self.action_dim,
            self.args.seed,
            theta=hyper_params["OU_NOISE_THETA"],
            sigma=hyper_params["OU_NOISE_SIGMA"],
        )

        # replay memory
        self.memory = ReplayBuffer(
            hyper_params["BUFFER_SIZE"], hyper_params["BATCH_SIZE"], self.args.seed
        )

        # load demo replay memory
        with open(self.args.demo_path, "rb") as f:
            demo = pickle.load(f)

        self.demo_memory = ReplayBuffer(
            len(demo), hyper_params["DEMO_BATCH_SIZE"], self.args.seed, demo
        )

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        self.curr_state = state

        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state)
        selected_action += torch.FloatTensor(self.noise.sample()).to(device)

        selected_action = torch.clamp(selected_action, -1.0, 1.0)

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)
        self.memory.add(self.curr_state, action, reward, next_state, done)

        return next_state, reward, done

    def update_model(
        self,
        experiences: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
        demos: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        exp_states, exp_actions, exp_rewards, exp_next_states, exp_dones = experiences
        demo_states, demo_actions, demo_rewards, demo_next_states, demo_dones = demos

        states = torch.cat((exp_states, demo_states), dim=0)
        actions = torch.cat((exp_actions, demo_actions), dim=0)
        rewards = torch.cat((exp_rewards, demo_rewards), dim=0)
        next_states = torch.cat((exp_next_states, demo_next_states), dim=0)
        dones = torch.cat((exp_dones, demo_dones), dim=0)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states, next_actions)
        curr_returns = rewards + (hyper_params["GAMMA"] * next_values * masks)
        curr_returns = curr_returns.to(device)

        # crittic loss
        values = self.critic(states, actions)
        critic_loss = F.mse_loss(values, curr_returns)

        # train critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # policy loss
        actions = self.actor(states)
        policy_loss = -self.critic(states, actions).mean()

        # bc loss
        pred_actions = self.actor(demo_states)
        qf_mask = torch.gt(
            self.critic(demo_states, demo_actions),
            self.critic(demo_states, pred_actions),
        ).to(device)
        qf_mask = qf_mask.float()
        bc_loss = F.mse_loss(
            torch.mul(pred_actions, qf_mask), torch.mul(demo_actions, qf_mask)
        )

        # train actor: pg loss + BC loss
        actor_loss = self.lambda1 * policy_loss + self.lambda2 * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        common_utils.soft_update(self.actor, self.actor_target, hyper_params["TAU"])
        common_utils.soft_update(self.critic, self.critic_target, hyper_params["TAU"])

        return actor_loss, critic_loss

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.actor_optimizer.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optimizer.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optimizer.state_dict(),
            "critic_optim_state_dict": self.critic_optimizer.state_dict(),
        }

        AbstractAgent.save_params(self, self.args.algo, params, n_episode)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor, self.critic], log="parameters")

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
                    experiences = self.memory.sample()
                    demos = self.demo_memory.sample()
                    loss = self.update_model(experiences, demos)
                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                total_loss = avg_loss.sum()
                print(
                    "[INFO] episode %d total score: %d, total loss: %f\n"
                    "actor_loss: %.3f critic_loss: %.3f"
                    % (
                        i_episode,
                        score,
                        total_loss,
                        avg_loss[0],  # actor loss
                        avg_loss[1],  # critic loss
                    )
                )

                if self.args.log:
                    wandb.log(
                        {
                            "score": score,
                            "total loss": total_loss,
                            "actor loss": avg_loss[0],
                            "critic loss": avg_loss[1],
                        }
                    )

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)
                if i_episode % self.args.save_period == 0:
                    self.save_params(i_episode)

        # termination
        self.env.close()
