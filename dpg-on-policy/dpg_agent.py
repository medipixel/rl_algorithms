# -*- coding: utf-8 -*-
"""DPG agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://proceedings.mlr.press/v32/silver14.pdf
"""

import os
import git
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb  # 'wandb off' in the shell makes this diabled


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DPGAgent(object):
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

    def select_action(self, state):
        """Select an action from the input space."""
        selected_action = self.actor(state)

        return selected_action

    def step(self, action):
        """Take an action and return the response of the env."""
        action = action.detach().to('cpu').numpy()
        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done

    def train(self, experience):
        """Train the model after each episode."""
        state, action, reward, next_state, done = experience

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        value = self.critic(state, action)
        next_action = self.actor(next_state)
        next_value = self.critic(next_state, next_action).detach()
        curr_return = reward + (self.args.gamma * next_value * (1 - done))
        curr_return = curr_return.to(device)

        # train critic
        critic_loss = F.mse_loss(value, curr_return)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        action = self.actor(state)
        actor_loss = -self.critic(state, action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

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
        path = os.path.join('./save/dpg_continuous_' +
                            sha[:7] +
                            '_ep_' +
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
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                loss = self.train((state, action, reward, next_state, done))

                state = next_state
                score += reward

                loss_episode.append(loss)  # for logging

            else:
                avg_loss = np.array(loss_episode).mean()
                print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                      % (i_episode, score, avg_loss))
                wandb.log({'score': score, 'avg_loss': avg_loss})

        # termination
        self.env.close()
        self.save_params(self.args.episode_num)
