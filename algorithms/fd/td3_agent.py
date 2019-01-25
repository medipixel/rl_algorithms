# -*- coding: utf-8 -*-
"""TD3 with PER using demo agent for episodic tasks in OpenAI Gym.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1802.09477.pdf
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

import algorithms.utils as common_utils
from algorithms.abstract_agent import AbstractAgent
from algorithms.noise import GaussianNoise
from algorithms.priortized_replay_buffer import PrioritizedReplayBuffer
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
    "MAX_EPISODE_STEPS": 300,
    "EPISODE_NUM": 1500,
    "PRETRAIN_NUM": 500,
    "N_STEP_RETURNS": 5,
    "PER_ALPHA": 0.5,
    "PER_BETA": 0.4,
    "PER_EPS": 1e-6,
}
DEMO_PATH = "data/lunarlander_continuous_demo.pkl"


class Agent(AbstractAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (ReplayBuffer): replay memory
        noise (OUNoise): random noise for exploration
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
        reward_list (list): list for n-step rewards

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

        # environment setup
        self.env._max_episode_steps = hyper_params["MAX_EPISODE_STEPS"]

        # create actor
        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_low, self.action_high
        ).to(device)
        self.actor_target = Actor(
            self.state_dim, self.action_dim, self.action_low, self.action_high
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # create critic
        self.critic_1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_2 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target2 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target1.load_state_dict(self.critic_1.state_dict())
        self.critic_target2.load_state_dict(self.critic_2.state_dict())

        # create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer1 = optim.Adam(self.critic_1.parameters(), lr=1e-3)
        self.critic_optimizer2 = optim.Adam(self.critic_2.parameters(), lr=1e-3)

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # noise instance to make randomness of action
        self.noise = GaussianNoise(self.action_dim, self.action_low, self.action_high)

        # load demo replay memory
        with open(DEMO_PATH, "rb") as f:
            demo = pickle.load(f)

        # replay memory
        self.beta = hyper_params["PER_BETA"]
        self.memory = PrioritizedReplayBuffer(
            hyper_params["BUFFER_SIZE"],
            hyper_params["BATCH_SIZE"],
            self.args.seed,
            demo=demo,
            alpha=hyper_params["PER_ALPHA"],
        )

        # n-step rewards
        # self.reward_list = list()

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        self.curr_state = state

        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state)

        action_size = selected_action.size()
        selected_action += torch.FloatTensor(
            self.noise.sample(action_size, self.n_step)
        ).to(device)

        return torch.clamp(selected_action, self.action_low, self.action_high)

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        self.state_list.append(self.state)
        self.action_list.append(self.action)
        self.next_state_list.append(next_state)
        self.reward_list.append(reward)
        self.done_list.append(done)

        if len(self.reward_list) > hyper_params["N_STEP_RETURNS"]:
            self.reward_list.pop(0)

        self.memory.add(self.curr_state, action, reward, next_state, done)

        return next_state, reward, done

    def rollout(self):
        rewards = list()
        action = None
        state = self.state
        n_step = hyper_params["N_STEP_RETURNS"]

        # for i in n_step - 1:
        #     action = self.select_action(state)
        #     state, reward, _ = self.n_step(action)
        #     rewards.append(reward * hyper_params["GAMMA"] ** i)

        next_values1 = self.critic_target1(state, action)
        next_values2 = self.critic_target2(state, action)
        next_values = torch.min(next_values1, next_values2)

        n_returns = sum(rewards) + next_values * hyper_params["GAMMA"] ** (n_step - 1)

        return n_returns

    def update_model(self, experiences: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones, weights, indexes = experiences
        masks = 1 - dones

        # get actions with noise
        next_actions = self.actor_target(next_states)
        noise = torch.normal(
            torch.zeros(next_actions.size()), hyper_params["NOISE_STD"]
        ).to(device)
        noise = torch.clamp(
            noise, -hyper_params["NOISE_CLIP"], hyper_params["NOISE_CLIP"]
        )
        next_actions += noise

        # min (Q_1', Q_2')
        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + hyper_params["GAMMA"] * next_values * masks
        curr_returns = curr_returns.to(device).detach()

        # n-step loss
        # n_returns = sum(self.reward_list)

        # critic loss
        values1 = self.critic_1(states, actions)
        values2 = self.critic_2(states, actions)
        critic_loss1 = F.mse_loss(values1, curr_returns) * torch.mean(weights)
        critic_loss2 = F.mse_loss(values2, curr_returns) * torch.mean(weights)

        # train critic
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        # for logging
        total_loss = critic_loss1 + critic_loss2

        # update target networks
        common_utils.soft_update(
            self.critic_1, self.critic_target1, hyper_params["TAU"]
        )
        common_utils.soft_update(
            self.critic_2, self.critic_target2, hyper_params["TAU"]
        )

        actor_loss = np.zeros(hyper_params["BATCH_SIZE"])
        if self.n_step % hyper_params["DELAYED_UPDATE"] == 0:
            # train actor
            actions = self.actor(states)
            actor_loss = -self.critic_1(states, actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            common_utils.soft_update(self.actor, self.actor_target, hyper_params["TAU"])

            # for logging
            total_loss += actor_loss

        # update priorities in PER
        new_priorities = (
            (torch.pow(torch.min(values1, values2) - curr_returns, 2) * weights)
            .data.cpu()
            .numpy()
        )
        new_priorities += np.power(actor_loss, 2)
        new_priorities += hyper_params["PER_EPS"]
        # new_priorities += EPS_D

        self.memory.update_priorities(indexes, new_priorities)

        return total_loss.data

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

        AbstractAgent.save_params(self, "td3", params, n_episode)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor, self.critic_1, self.critic_2], log="parameters")

        loss_episode = list()
        # pre-training by demo
        self.n_step = 0
        for _ in range(1, hyper_params["PRETRAIN_NUM"] + 1):
            experiences = self.memory.sample()
            loss = self.update_model(experiences)
            loss_episode.append(loss)  # for logging
            self.n_step += 1

        # train
        self.n_step = 0
        for i_episode in range(1, hyper_params["EPISODE_NUM"] + 1):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.n_step(action)

                # n_returns = self.rollout(state)

                if len(self.memory) >= hyper_params["BATCH_SIZE"]:
                    experiences = self.memory.sample()
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward
                self.n_step += 1

            if loss_episode:
                avg_loss = np.array(loss_episode).mean()
                print(
                    "[INFO] episode %d\ttotal score: %d\tloss: %f"
                    % (i_episode, score, avg_loss)
                )

                if self.args.log:
                    wandb.log({"score": score, "avg_loss": avg_loss})

                if i_episode % self.args.save_period == 0:
                    self.save_params(i_episode)

        # termination
        self.env.close()
