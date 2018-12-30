# -*- coding: utf-8 -*-
"""TRPO agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://arxiv.org/abs/1502.05477
"""

import os
import git
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils

import wandb  # 'wandb off' in the shell makes this diabled


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TRPOAgent(object):
    """ActorCritic interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment with discrete action space
        model (nn.Module): policy gradient model to select actions
        args (dict): arguments including hyperparameters and training settings

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        model (nn.Module): policy gradient model to select actions
        args (dict): arguments including hyperparameters and training settings
        optimizer (Optimizer): optimizer for training

    """

    def __init__(self, env, actor, critic, args):
        """Initialization."""
        assert(issubclass(type(env), gym.Env))
        assert(issubclass(type(actor), nn.Module))
        assert(issubclass(type(critic), nn.Module))

        self.env = env
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(actor.parameters())
        self.critic_optimizer = optim.Adam(critic.parameters())
        if args.model_path is not None and os.path.exists(args.model_path):
            self.load_params(args.model_path)
        self.args = args
        self.memory = deque()

    def select_action(self, state):
        """Select an action from the input space."""
        selected_action, _, _ = self.actor(state)

        return selected_action

    def step(self, state, action):
        """Take an action and return the response of the env."""
        action = action.detach().to('cpu').numpy()
        next_state, reward, done, _ = self.env.step(action)
        self.memory.append([state, action, reward, done])

        return next_state, reward, done

    def train(self):
        """Train the model after every N episodes."""
        states, actions, rewards, dones = utils.decompose_memory(self.memory)
        values = self.critic(states)

        returns, advantages = utils.get_ret_and_gae(rewards, values, dones,
                                                    self.args.gamma,
                                                    self.args.lambd)

        # normalize the advantages
        advantages = (advantages - advantages.mean()) /\
                     (advantages.std() + 1e-7)

        # train actor
        actor_loss = utils.trpo_step(self.actor, states, advantages,
                                     self.args.max_kl, self.args.damping)

        # train critic
        critic_loss = F.mse_loss(values, returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # for logging
        total_loss = critic_loss + actor_loss

        return total_loss.data

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print('[ERROR] the input path does not exist. ->', path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params['actor_state_dict'])
        self.critic.load_state_dict(params['critic_state_dict'])
        self.actor_optimizer.load_state_dict(
                params['actor_optim_state_dict'])
        self.critic_optimizer.load_state_dict(
                params['critic_optim_state_dict'])
        print('[INFO] loaded the model and optimizer from', path)

    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        if not os.path.exists('./save'):
            os.mkdir('./save')

        params = {
                 'actor_state_dict':
                 self.actor.state_dict(),
                 'critic_state_dict':
                 self.critic.state_dict(),
                 'actor_optim_state_dict':
                 self.actor_optimizer.state_dict(),
                 'critic_optim_state_dict':
                 self.critic_optimizer.state_dict()
                 }

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        path = os.path.join('./save/trpo_' + sha[:7] + '_ep_' +
                            str(n_episode)+'.pt')
        torch.save(params, path)
        print('[INFO] saved the model and optimizer to', path)

    def run(self):
        """Run the agent."""
        # logger
        wandb.init()
        wandb.config.update(self.args)
        wandb.watch(self.actor, log='parameters')
        wandb.watch(self.critic, log='parameters')

        for i_episode in range(self.args.episode_num):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(state, action)

                state = next_state
                score += reward

            # train every self.args.epicodes_per_batch
            if i_episode % self.args.episodes_per_batch == 0:
                loss = self.train()
                print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                      % (i_episode, score, loss))
                wandb.log({'score': score, 'loss': loss})

        # termination
        self.env.close()
        self.save_params(self.args.episode_num)
