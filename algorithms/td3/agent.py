# -*- coding: utf-8 -*-
"""TD3 agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1802.09477.pdf
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb

import algorithms.common.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.buffer.replay_buffer import ReplayBuffer
from algorithms.common.noise import GaussianNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (ReplayBuffer): replay memory
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
        hyper_params (dict): hyper-parameters
        curr_state (np.ndarray): temporary storage of the current state
        n_step (int): iteration number of the current episode

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
        noise: GaussianNoise,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic
            noise (GaussianNoise): random noise for exploration

        """
        AbstractAgent.__init__(self, env, args)

        self.actor, self.actor_target = models[0:2]
        self.critic_1, self.critic_2 = models[2:4]
        self.critic_target1, self.critic_target2 = models[4:6]
        self.actor_optimizer = optims[0]
        self.critic_optimizer1, self.critic_optimizer2 = optims[1:3]
        self.hyper_params = hyper_params
        self.curr_state = np.zeros((1,))
        self.noise = noise
        self.n_step = 0

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # replay memory
        self.memory = ReplayBuffer(
            hyper_params["BUFFER_SIZE"], hyper_params["BATCH_SIZE"]
        )

    def select_action(self, state: np.ndarray) -> torch.Tensor:
        """Select an action from the input space."""
        self.curr_state = state

        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state)

        if not self.args.test:
            action_size = selected_action.size()
            selected_action += torch.FloatTensor(
                self.noise.sample(action_size, self.n_step)
            ).to(device)
            selected_action = torch.clamp(selected_action, -1.0, 1.0)

        return selected_action

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = self.env.step(action)

        self.memory.add(self.curr_state, action, reward, next_state, float(done))

        return next_state, reward, done

    def update_model(
        self,
        experiences: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones = experiences
        masks = 1 - dones

        # get actions with noise
        noise_std, noise_clip = (
            self.hyper_params["TARGET_SMOOTHING_NOISE_STD"],
            self.hyper_params["TARGET_SMOOTHING_NOISE_CLIP"],
        )
        next_actions = self.actor_target(next_states)
        noise = torch.normal(torch.zeros(next_actions.size()), noise_std).to(device)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_actions += noise
        next_actions = torch.clamp(next_actions, -1.0, 1.0)

        # min (Q_1', Q_2')
        next_states_actions = torch.cat((next_states, next_actions), dim=-1)
        next_values1 = self.critic_target1(next_states_actions)
        next_values2 = self.critic_target2(next_states_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + self.hyper_params["GAMMA"] * next_values * masks
        curr_returns = curr_returns.to(device).detach()

        # critic loss
        states_actions = torch.cat((states, actions), dim=-1)
        values1 = self.critic_1(states_actions)
        values2 = self.critic_2(states_actions)
        critic_loss1 = F.mse_loss(values1, curr_returns)
        critic_loss2 = F.mse_loss(values2, curr_returns)

        # train critic
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if self.n_step % self.hyper_params["DELAYED_UPDATE"] == 0:
            # train actor
            actions = self.actor(states)
            states_actions = torch.cat((states, actions), dim=-1)
            actor_loss = -self.critic_1(states_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            tau = self.hyper_params["TAU"]
            common_utils.soft_update(self.critic_1, self.critic_target1, tau)
            common_utils.soft_update(self.critic_2, self.critic_target2, tau)
            common_utils.soft_update(self.actor, self.actor_target, tau)
        else:
            actor_loss = torch.zeros(1)

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

        AbstractAgent.save_params(self, params, n_episode)

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
            wandb.config.update(self.hyper_params)
            wandb.watch([self.actor, self.critic_1, self.critic_2], log="parameters")

        self.n_step = 0
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

                state = next_state
                score += reward
                self.n_step += 1

            # training
            if len(self.memory) >= self.hyper_params["BATCH_SIZE"]:
                for _ in range(self.hyper_params["EPOCH"]):
                    experiences = self.memory.sample()
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(
                    i_episode, avg_loss, score, self.hyper_params["DELAYED_UPDATE"]
                )
            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
