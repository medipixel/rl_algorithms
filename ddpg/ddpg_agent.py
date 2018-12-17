# -*- coding: utf-8 -*-
"""DDPG agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
"""

import os
import git
import random
import copy
import numpy as np
from collections import namedtuple, deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb  # 'wandb off' in the shell makes this diabled


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DDPGAgent(object):
    """ActorCritic interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment with continuous action space
        actor_local (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic_local (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic
        args (dict): arguments including hyperparameters and training settings
        noise (OUNoise): random noise for exploration
        memory (ReplayBuffer): replay memory

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        actor_local (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic_local (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        args (dict): arguments including hyperparameters and training settings

    """

    def __init__(self, env, actor_local, actor_target,
                 critic_local, critic_target, args):
        """Initialization."""
        assert(issubclass(type(env), gym.Env))
        assert(issubclass(type(actor_local), nn.Module))
        assert(issubclass(type(actor_target), nn.Module))
        assert(issubclass(type(critic_local), nn.Module))
        assert(issubclass(type(critic_target), nn.Module))

        self.env = env

        # models and optimizers
        self.actor_local = actor_local
        self.actor_target = actor_target
        self.critic_local = critic_local
        self.critic_target = critic_target
        self.actor_optimizer = optim.Adam(actor_local.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(critic_local.parameters(), lr=1e-3)

        # load the optimizer and model parameters
        if args.model_path is not None and os.path.exists(args.model_path):
            self.load_params(args.model_path)

        # arguments
        self.args = args

        # noise instance to make randomness of action
        action_dim = self.env.action_space.shape[0]
        self.noise = OUNoise(action_dim,
                             self.args.seed)

        # replay memory
        self.memory = ReplayBuffer(action_dim,
                                   self.args.buffer_size,
                                   self.args.batch_size,
                                   self.args.seed)

    def select_action(self, state):
        """Select an action from the input space."""
        selected_action = self.actor_local(state)
        selected_action += torch.tensor(self.noise.sample()).float().to(device)

        action_low = float(self.env.action_space.low[0])
        action_high = float(self.env.action_space.high[0])
        selected_action = torch.clamp(selected_action, action_low, action_high)

        return selected_action

    def step(self, state, action):
        """Take an action and return the response of the env."""
        action = action.detach().to('cpu').numpy()
        next_state, reward, done, _ = self.env.step(action)
        self.memory.add(state, action, reward, next_state, done)

        return next_state, reward, done

    def train(self, experiences):
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones = experiences

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states, next_actions)
        curr_returns = rewards + (self.args.gamma * next_values * (1 - dones))
        curr_returns = curr_returns.to(device)

        # train critic
        values = self.critic_local(states, actions)
        critic_loss = F.mse_loss(values, curr_returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # train actor
        actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)

        # for logging
        total_loss = critic_loss + actor_loss

        return total_loss.data

    def soft_update(self, local, target):
        """Soft-update: target = tau*local + (1-tau)*target."""
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(self.args.tau*l_param.data +
                               (1.0-self.args.tau)*t_param.data)

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print('[ERROR] the input path does not exist. ->', path)
            return

        params = torch.load(path)
        self.actor_local.load_state_dict(params['actor_local_state_dict'])
        self.actor_target.load_state_dict(params['actor_target_state_dict'])
        self.critic_local.load_state_dict(params['critic_local_state_dict'])
        self.critic_target.load_state_dict(params['critic_target_state_dict'])
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
                 'actor_local_state_dict':
                 self.actor_local.state_dict(),
                 'actor_target_state_dict':
                 self.actor_target.state_dict(),
                 'critic_local_state_dict':
                 self.critic_local.state_dict(),
                 'critic_target_state_dict':
                 self.critic_target.state_dict(),
                 'actor_optim_state_dict':
                 self.actor_optimizer.state_dict(),
                 'critic_optim_state_dict':
                 self.critic_optimizer.state_dict()
                 }

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        path = os.path.join('./save/ddpg_' + sha[:7] + '_ep_' +
                            str(n_episode)+'.pt')
        torch.save(params, path)
        print('[INFO] saved the model and optimizer to', path)

    def run(self):
        """Run the agent."""
        # logger
        wandb.init()
        wandb.config.update(self.args)
        wandb.watch([self.actor_local, self.critic_local], log='parameters')

        for i_episode in range(self.args.episode_num):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)

                next_state, reward, done = self.step(state, action)

                if len(self.memory) >= self.args.batch_size:
                    experiences = self.memory.sample()
                    loss = self.train(experiences)

                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward

            else:
                avg_loss = np.array(loss_episode).mean()
                print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                      % (i_episode, score, avg_loss))
                wandb.log({'score': score, 'avg_loss': avg_loss})

        # termination
        self.env.close()
        self.save_params(self.args.episode_num)


class OUNoise:
    """Ornstein-Uhlenbeck process.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) +\
            self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for e in experiences:
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            dones.append(e.done)

        states =\
            torch.from_numpy(np.vstack(states)).float().to(device)
        actions =\
            torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards =\
            torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states =\
            torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones =\
            torch.from_numpy(
                np.vstack(dones).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
