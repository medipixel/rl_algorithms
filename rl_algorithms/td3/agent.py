# -*- coding: utf-8 -*-
"""TD3 agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1802.09477.pdf
"""

import argparse
import time
from typing import Tuple

import gym
import numpy as np
from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.buffer.replay_buffer import ReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.networks.mlp import MLP, FlattenMLP
from rl_algorithms.common.noise import GaussianNoise
from rl_algorithms.registry import AGENTS
from rl_algorithms.utils.config import ConfigDict
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class TD3Agent(Agent):
    """ActorCritic interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        memory (ReplayBuffer): replay memory
        exploration_noise (GaussianNoise): random noise for exploration
        target_policy_noise (GaussianNoise): random noise for target values
        actor (nn.Module): actor model to select actions
        critic1 (nn.Module): critic model to predict state values
        critic2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        actor_target (nn.Module): target actor model to select actions
        critic_optim (Optimizer): optimizer for training critic
        actor_optim (Optimizer): optimizer for training actor
        curr_state (np.ndarray): temporary storage of the current state
        total_steps (int): total step numbers
        episode_steps (int): step number of the current episode
        i_episode (int): current episode number
        noise_cfg (ConfigDict): config of noise

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        network_cfg: ConfigDict,
        optim_cfg: ConfigDict,
        noise_cfg: ConfigDict,
    ):
        """Initialize.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings

        """
        Agent.__init__(self, env, args, log_cfg)

        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.episode_step = 0
        self.update_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.noise_cfg = noise_cfg
        self.network_cfg = network_cfg
        self.optim_cfg = optim_cfg

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        # noise instance to make randomness of action
        self.exploration_noise = GaussianNoise(
            self.action_dim, noise_cfg.exploration_noise, noise_cfg.exploration_noise
        )

        self.target_policy_noise = GaussianNoise(
            self.action_dim,
            noise_cfg.target_policy_noise,
            noise_cfg.target_policy_noise,
        )

        if not self.args.test:
            # replay memory
            self.memory = ReplayBuffer(
                self.hyper_params.buffer_size, self.hyper_params.batch_size
            )

        self._init_network()

    def _init_network(self):
        """Initialize networks and optimizers."""
        # create actor
        self.actor = MLP(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=self.network_cfg.hidden_sizes_actor,
            output_activation=torch.tanh,
        ).to(device)

        self.actor_target = MLP(
            input_size=self.state_dim,
            output_size=self.action_dim,
            hidden_sizes=self.network_cfg.hidden_sizes_actor,
            output_activation=torch.tanh,
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # create critic
        self.critic1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_critic,
        ).to(device)

        self.critic2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_critic,
        ).to(device)

        self.critic_target1 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_critic,
        ).to(device)

        self.critic_target2 = FlattenMLP(
            input_size=self.state_dim + self.action_dim,
            output_size=1,
            hidden_sizes=self.network_cfg.hidden_sizes_critic,
        ).to(device)

        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # concat critic parameters to use one optim
        critic_parameters = list(self.critic1.parameters()) + list(
            self.critic2.parameters()
        )

        # create optimizers
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.critic_optim = optim.Adam(
            critic_parameters,
            lr=self.optim_cfg.lr_critic,
            weight_decay=self.optim_cfg.weight_decay,
        )

        # load the optimizer and model parameters
        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        # initial training step, try random action for exploration
        self.curr_state = state

        if (
            self.total_step < self.hyper_params.initial_random_action
            and not self.args.test
        ):
            return np.array(self.env.action_space.sample())

        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state).detach().cpu().numpy()

        if not self.args.test:
            noise = self.exploration_noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            # if last state is not terminal state in episode, done is false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            self.memory.add((self.curr_state, action, reward, next_state, done_bool))

        return next_state, reward, done, info

    def update_model(self) -> Tuple[torch.Tensor, ...]:
        """Train the model after each episode."""
        self.update_step += 1

        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        masks = 1 - dones

        # get actions with noise
        noise = torch.FloatTensor(self.target_policy_noise.sample()).to(device)
        clipped_noise = torch.clamp(
            noise,
            -self.noise_cfg.target_policy_noise_clip,
            self.noise_cfg.target_policy_noise_clip,
        )
        next_actions = (self.actor_target(next_states) + clipped_noise).clamp(-1.0, 1.0)

        # min (Q_1', Q_2')
        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + self.hyper_params.gamma * next_values * masks
        curr_returns = curr_returns.detach()

        # critic loss
        values1 = self.critic1(states, actions)
        values2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(values1, curr_returns)
        critic2_loss = F.mse_loss(values2, curr_returns)

        # train critic
        critic_loss = critic1_loss + critic2_loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.update_step % self.hyper_params.policy_update_freq == 0:
            # policy loss
            actions = self.actor(states)
            actor_loss = -self.critic1(states, actions).mean()

            # train actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            tau = self.hyper_params.tau
            common_utils.soft_update(self.critic1, self.critic_target1, tau)
            common_utils.soft_update(self.critic2, self.critic_target2, tau)
            common_utils.soft_update(self.actor, self.actor_target, tau)
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path)
        self.critic1.load_state_dict(params["critic1"])
        self.critic2.load_state_dict(params["critic2"])
        self.critic_target1.load_state_dict(params["critic_target1"])
        self.critic_target2.load_state_dict(params["critic_target2"])
        self.critic_optim.load_state_dict(params["critic_optim"])
        self.actor.load_state_dict(params["actor"])
        self.actor_target.load_state_dict(params["actor_target"])
        self.actor_optim.load_state_dict(params["actor_optim"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):  # type: ignore
        """Save model and optimizer parameters."""
        params = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic_target1": self.critic_target1.state_dict(),
            "critic_target2": self.critic_target2.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
        }

        Agent.save_params(self, params, n_episode)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, policy_update_freq, avg_time_cost = log_value
        total_loss = loss.sum()
        print(
            "[INFO] episode %d, episode_step: %d, total_step: %d, total score: %d\n"
            "total loss: %f actor_loss: %.3f critic1_loss: %.3f critic2_loss: %.3f "
            "(spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0] * policy_update_freq,  # actor loss
                loss[1],  # critic1 loss
                loss[2],  # critic2 loss
                avg_time_cost,
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * policy_update_freq,
                    "critic1 loss": loss[1],
                    "critic2 loss": loss[2],
                    "time per each step": avg_time_cost,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.actor, self.critic1, self.critic2], log="parameters")

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()
            self.episode_step = 0

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                state = next_state
                score += reward

                if len(self.memory) >= self.hyper_params.batch_size:
                    loss = self.update_model()
                    loss_episode.append(loss)  # for logging

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                log_value = (
                    self.i_episode,
                    avg_loss,
                    score,
                    self.hyper_params.policy_update_freq,
                    avg_time_cost,
                )
                self.write_log(log_value)
            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.save_params(self.i_episode)
        self.interim_test()
