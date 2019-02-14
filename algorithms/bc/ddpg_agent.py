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
import wandb

import algorithms.common.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.buffer.replay_buffer import ReplayBuffer
from algorithms.common.noise import OUNoise
from algorithms.her import HER

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """ActorCritic interacting with environment.

    Attributes:
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        hyper_params (dict): hyper-parameters
        noise (OUNoise): random noise for exploration
        memory (ReplayBuffer): replay memory
        curr_state (np.ndarray): temporary storage of the current state
        her (HER): hinsight experience replay
        transitions_epi (list): transitions per episode (for HER)
        goal_state (np.ndarray): goal state to generate concatenated states
        total_step (int): total step numbers
        episode_step (int): step number of the current episode

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
        noise: OUNoise,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic
            noise (OUNoise): random noise for exploration

        """
        AbstractAgent.__init__(self, env, args)

        self.actor, self.actor_target, self.critic, self.critic_target = models
        self.actor_optimizer, self.critic_optimizer = optims
        self.hyper_params = hyper_params
        self.curr_state = np.zeros((1,))
        self.noise = noise
        self.total_step = 0
        self.episode_step = 0

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # load demo replay memory
        with open(self.args.demo_path, "rb") as f:
            demo = list(pickle.load(f))

        # HER
        if hyper_params["USE_HER"]:
            self.her = HER(self.args.demo_path)
            self.transitions_epi: list = list()
            self.desired_state = np.zeros((1,))
            demo = self.her.generate_demo_transitions(demo)

        if not self.args.test:
            # Replay buffers
            self.demo_memory = ReplayBuffer(len(demo), hyper_params["DEMO_BATCH_SIZE"])
            self.demo_memory.extend(demo)

            self.memory = ReplayBuffer(
                hyper_params["BUFFER_SIZE"], hyper_params["BATCH_SIZE"]
            )

            # set hyper parameters
            self.lambda1 = hyper_params["LAMBDA1"]
            self.lambda2 = hyper_params["LAMBDA2"] / hyper_params["DEMO_BATCH_SIZE"]

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # HER
        if self.hyper_params["USE_HER"]:
            self.desired_state = self.her.sample_desired_state()
            state = np.concatenate((state, self.desired_state), axis=-1)

        # if initial random action should be conducted
        if (
            self.total_step < self.hyper_params["INITIAL_RANDOM_ACTION"]
            and not self.args.test
        ):
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state)

        if not self.args.test:
            selected_action += torch.FloatTensor(self.noise.sample()).to(device)
            selected_action = torch.clamp(selected_action, -1.0, 1.0)

        return selected_action.detach().cpu().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        self.total_step += 1
        self.episode_step += 1

        next_state, reward, done, _ = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            e = (self.curr_state, action, reward, next_state, done_bool)

            # HER
            if self.hyper_params["USE_HER"]:
                self.transitions_epi.append(e)

                # insert generated transitions if the episode is done
                if done:
                    transitions = self.her.generate_transitions(
                        self.transitions_epi, self.desired_state
                    )
                    self.memory.extend(transitions)
            else:
                self.memory.add(*e)

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
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + (self.hyper_params["GAMMA"] * next_values * masks)
        curr_returns = curr_returns.to(device)

        # critic loss
        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)

        # train critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # policy loss
        actions = self.actor(states)
        policy_loss = -self.critic(torch.cat((states, actions), dim=-1)).mean()

        # bc loss
        pred_actions = self.actor(demo_states)
        qf_mask = torch.gt(
            self.critic(torch.cat((demo_states, demo_actions), dim=-1)),
            self.critic(torch.cat((demo_states, pred_actions), dim=-1)),
        ).to(device)
        qf_mask = qf_mask.float()
        n_qf_mask = int(qf_mask.sum().item())

        if n_qf_mask == 0:
            bc_loss = torch.zeros(1, device=device)
        else:
            bc_loss = (
                torch.mul(pred_actions, qf_mask) - torch.mul(demo_actions, qf_mask)
            ).pow(2).sum() / n_qf_mask

        # train actor: pg loss + BC loss
        actor_loss = self.lambda1 * policy_loss + self.lambda2 * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        tau = self.hyper_params["TAU"]
        common_utils.soft_update(self.actor, self.actor_target, tau)
        common_utils.soft_update(self.critic, self.critic_target, tau)

        return actor_loss.data, critic_loss.data

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

        AbstractAgent.save_params(self, params, n_episode)

    def write_log(self, i: int, loss: np.ndarray, score: float = 0.0):
        """Write log about loss and score"""
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "total loss: %f actor_loss: %.3f critic_loss: %.3f\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0],
                loss[1],
            )  # actor loss  # critic loss
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0],
                    "critic loss": loss[1],
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            wandb.watch([self.actor, self.critic], log="parameters")

        for i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                if len(self.memory) >= self.hyper_params["BATCH_SIZE"]:
                    experiences = self.memory.sample()
                    demos = self.demo_memory.sample()
                    loss = self.update_model(experiences, demos)
                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(i_episode, avg_loss, score)

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
