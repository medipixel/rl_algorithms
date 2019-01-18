# -*- coding: utf-8 -*-
"""DDPG agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
"""

import os
import git
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb

from baselines.ddpg.model import Actor, Critic
from baselines.noise import OUNoise
from baselines.replay_buffer import ReplayBuffer


# hyper parameters
hyper_params = {
        'GAMMA': 0.99,
        'TAU': 1e-3,
        'BUFFER_SIZE': int(1e5),
        'BATCH_SIZE': 128,
        'MAX_EPISODE_STEPS': 300,
        'EPISODE_NUM': 1500
}


class Agent(object):
    """ActorCritic interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment with continuous action space
        args (dict): arguments including hyperparameters and training settings
        memory (ReplayBuffer): replay memory
        noise (OUNoise): random noise for exploration
        device (str): device selection (cpu / gpu)
        actor_local (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic_local (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        args (dict): arguments including hyperparameters and training settings
        device (str): device selection (cpu / gpu)

    """

    def __init__(self, env, args, device):
        """Initialization."""
        self.env = env
        self.args = args
        self.device = device
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = float(self.env.action_space.low[0])
        action_high = float(self.env.action_space.high[0])

        # environment setup
        self.env._max_episode_steps = hyper_params['MAX_EPISODE_STEPS']

        # create actor
        self.actor_local = Actor(state_dim, action_dim, action_low,
                                 action_high, self.device).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_low,
                                  action_high, self.device).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())

        # create critic
        self.critic_local = Critic(state_dim, action_dim,
                                   self.device).to(device)
        self.critic_target = Critic(state_dim, action_dim,
                                    self.device).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        # create optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=1e-3)

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # noise instance to make randomness of action
        self.noise = OUNoise(action_dim, self.args.seed,
                             theta=0., sigma=0.)

        # replay memory
        self.memory = ReplayBuffer(action_dim,
                                   hyper_params['BUFFER_SIZE'],
                                   hyper_params['BATCH_SIZE'],
                                   self.args.seed, self.device)

    def select_action(self, state):
        """Select an action from the input space."""
        selected_action = self.actor_local(state)
        selected_action += torch.tensor(self.noise.sample()
                                        ).float().to(self.device)

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

    def update_model(self, experiences):
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones = experiences

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states, next_actions)
        curr_returns = rewards + (hyper_params['GAMMA'] * next_values * masks)
        curr_returns = curr_returns.to(self.device)

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
            t_param.data.copy_(hyper_params['TAU']*l_param.data +
                               (1.0-hyper_params['TAU'])*t_param.data)

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

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor_local, self.critic_local],
                        log='parameters')

        for i_episode in range(1, hyper_params['EPISODE_NUM']+1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(state, action)
                if len(self.memory) >= hyper_params['BATCH_SIZE']:
                    experiences = self.memory.sample()
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward

            else:
                avg_loss = np.array(loss_episode).mean()
                print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                      % (i_episode, score, avg_loss))

                if self.args.log:
                    wandb.log({'score': score, 'avg_loss': avg_loss})

        # termination
        self.env.close()
        self.save_params(hyper_params['EPISODE_NUM'])

    def test(self):
        """Test the agent."""
        for i_episode in range(hyper_params['EPISODE_NUM']):
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

            else:
                print('[INFO] episode %d\ttotal score: %d'
                      % (i_episode, score))

        # termination
        self.env.close()
