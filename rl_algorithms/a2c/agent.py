# -*- coding: utf-8 -*-
"""1-Step Advantage Actor-Critic agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
from typing import Tuple

import gym
import numpy as np
import torch
import torch.optim as optim
import wandb

from rl_algorithms.a2c.learner import A2CLearner
from rl_algorithms.common.abstract.agent import Agent
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.registry import AGENTS
from rl_algorithms.utils.config import ConfigDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class A2CAgent(Agent):
    """1-Step Advantage Actor-Critic interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        network_cfg (ConfigDict): config of network for training agent
        optim_cfg (ConfigDict): config of optimizer
        state_dim (int): state size of env
        action_dim (int): action size of env
        actor (nn.Module): policy model to select actions
        critic (nn.Module): critic model to evaluate states
        actor_optim (Optimizer): optimizer for actor
        critic_optim (Optimizer): optimizer for critic
        episode_step (int): step number of the current episode
        i_episode (int): current episode number
        transition (list): recent transition information

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        log_cfg: ConfigDict,
        hyper_params: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
        optim_cfg: ConfigDict,
    ):
        """Initialize."""
        Agent.__init__(self, env, args, log_cfg)

        self.transition: list = list()
        self.episode_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.backbone_cfg = backbone
        self.head_cfg = head
        self.optim_cfg = optim_cfg

        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]

        self._init_network()

    def _init_network(self):
        # create models

        self.head_cfg.actor.configs.state_size = (
            self.head_cfg.critic.configs.state_size
        ) = self.state_dim
        self.head_cfg.actor.configs.output_size = self.action_dim

        self.actor = Brain(self.backbone_cfg.actor, self.head_cfg.actor).to(device)
        self.critic = Brain(self.backbone_cfg.critic, self.head_cfg.critic).to(device)

        # create optimizer
        self.actor_optim = optim.Adam(
            self.actor.parameters(),
            lr=self.optim_cfg.lr_actor,
            weight_decay=self.optim_cfg.weight_decay,
        )

        self.critic_optim = optim.Adam(
            self.critic.parameters(),
            lr=self.optim_cfg.lr_critic,
            weight_decay=self.optim_cfg.weight_decay,
        )

        if self.args.load_from is not None:
            self.load_params(self.args.load_from)

        self.learner = A2CLearner(self.args, self.hyper_params, device)

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

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""

        action = action.detach().cpu().numpy()
        next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            done_bool = done
            if self.episode_step == self.args.max_episode_steps:
                done_bool = False
            self.transition.extend([next_state, reward, done_bool])

        return next_state, reward, done, info

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        Agent.load_params(self, path)

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] Loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
        }

        Agent._save_params(self, params, n_episode)

    def write_log(self, log_value: tuple):
        i, score, policy_loss, value_loss = log_value
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
            self.set_wandb()
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
                next_state, reward, done, _ = self.step(action)
                self.episode_step += 1

                policy_loss, value_loss = self.learner.update_model(
                    (self.actor, self.critic),
                    (self.actor_optim, self.critic_optim),
                    self.transition,
                )

                policy_loss_episode.append(policy_loss)
                value_loss_episode.append(value_loss)

                state = next_state
                score += reward

            # logging
            policy_loss = np.array(policy_loss_episode).mean()
            value_loss = np.array(value_loss_episode).mean()
            log_value = (self.i_episode, score, policy_loss, value_loss)
            self.write_log(log_value)

            if self.i_episode % self.args.save_period == 0:
                self.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.save_params(self.i_episode)
        self.interim_test()
