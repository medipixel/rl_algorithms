# -*- coding: utf-8 -*-
"""TD3 agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1802.09477.pdf
"""

import os
import git
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb

from algorithms.td3.model import Actor, Critic
from algorithms.noise import GaussianNoise
from algorithms.replay_buffer import ReplayBuffer


# hyper parameters
hyper_params = {
        'GAMMA': 0.99,
        'TAU': 1e-3,
        'NOISE_STD': 1.0,
        'NOISE_CLIP': 0.5,
        'DELAYED_UPDATE': 2,
        'BUFFER_SIZE': int(1e5),
        'BATCH_SIZE': 128,
        'MAX_EPISODE_STEPS': 300,
        'EPISODE_NUM': 1500
}


class Agent(object):
    """ActorCritic interacting with environment.

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        args (dict): arguments including hyperparameters and training settings
        device (torch.device): device selection (cpu / gpu)

    Attrtibutes:
        env (gym.Env): openAI Gym environment with continuous action space
        args (dict): arguments including hyperparameters and training settings
        memory (ReplayBuffer): replay memory
        noise (OUNoise): random noise for exploration
        action_low (float): lower bound of action values
        action_high (float): upper bound of action values
        device (str): device selection (cpu / gpu)
        actor_local (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic_local (nn.Module): critic model to predict state values
        critic_target (nn.Module): target critic model to predict state values
        actor_optimizer (Optimizer): optimizer for training actor
        critic_optimizer (Optimizer): optimizer for training critic

    """

    def __init__(self, env, args, device):
        """Initialization."""
        self.env = env
        self.args = args
        self.device = device
        self.action_low = float(self.env.action_space.low[0])
        self.action_high = float(self.env.action_space.high[0])
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # environment setup
        self.env._max_episode_steps = hyper_params['MAX_EPISODE_STEPS']

        # create actor
        self.actor_local = Actor(state_dim, action_dim, self.action_low,
                                 self.action_high, self.device).to(device)
        self.actor_target = Actor(state_dim, action_dim, self.action_low,
                                  self.action_high, self.device).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())

        # create critic
        self.critic_local1 = Critic(state_dim, action_dim,
                                    self.device).to(device)
        self.critic_local2 = Critic(state_dim, action_dim,
                                    self.device).to(device)
        self.critic_target1 = Critic(state_dim, action_dim,
                                     self.device).to(device)
        self.critic_target2 = Critic(state_dim, action_dim,
                                     self.device).to(device)
        self.critic_target1.load_state_dict(self.critic_local1.state_dict())
        self.critic_target2.load_state_dict(self.critic_local2.state_dict())

        # create optimizers
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=1e-4)
        self.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(),
                                            lr=1e-3)
        self.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(),
                                            lr=1e-3)

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        # noise instance to make randomness of action
        self.noise = GaussianNoise(action_dim,
                                   self.action_low, self.action_high)

        # replay memory
        self.memory = ReplayBuffer(hyper_params['BUFFER_SIZE'],
                                   hyper_params['BATCH_SIZE'],
                                   self.args.seed, self.device)

    def select_action(self, state, t):
        """Select an action from the input space."""
        selected_action = self.actor_local(state)
        action_size = selected_action.size()
        selected_action += torch.tensor(self.noise.sample(action_size, t)
                                        ).float().to(self.device)

        return torch.clamp(selected_action, self.action_low, self.action_high)

    def step(self, state, action):
        """Take an action and return the response of the env."""
        action = action.detach().to('cpu').numpy()
        next_state, reward, done, _ = self.env.step(action)
        self.memory.add(state, action, reward, next_state, done)

        return next_state, reward, done

    def update_model(self, experiences, step):
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones = experiences
        masks = 1 - dones

        # get actions with noise
        next_actions = self.actor_target(next_states)
        noise = torch.normal(torch.zeros(next_actions.size()),
                             hyper_params['NOISE_STD']).to(self.device)
        noise = torch.clamp(noise,
                            -hyper_params['NOISE_CLIP'],
                            hyper_params['NOISE_CLIP'])
        next_actions += noise

        # min (Q_1', Q_2')
        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + (hyper_params['GAMMA'] * next_values * masks)
        curr_returns = curr_returns.to(self.device).detach()

        # critic loss
        values1 = self.critic_local1(states, actions)
        values2 = self.critic_local2(states, actions)
        critic_loss1 = F.mse_loss(values1, curr_returns)
        critic_loss2 = F.mse_loss(values2, curr_returns)

        # train critic
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        # for logging
        total_loss = critic_loss1 + critic_loss2

        if step % hyper_params['DELAYED_UPDATE'] == 0:
            # train actor
            actions = self.actor_local(states)
            actor_loss = -self.critic_local1(states, actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            self.soft_update(self.critic_local1, self.critic_target1)
            self.soft_update(self.critic_local2, self.critic_target2)
            self.soft_update(self.actor_local, self.actor_target)

            # for logging
            total_loss += actor_loss

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
        self.critic_local1.load_state_dict(params['critic_local1'])
        self.critic_local2.load_state_dict(params['critic_local2'])
        self.critic_target1.load_state_dict(params['critic_target1'])
        self.critic_target2.load_state_dict(params['critic_target2'])
        self.critic_optimizer1.load_state_dict(params['critic_optim1'])
        self.critic_optimizer2.load_state_dict(params['critic_optim2'])
        self.actor_local.load_state_dict(params['actor_local'])
        self.actor_target.load_state_dict(params['actor_target'])
        self.actor_optimizer.load_state_dict(params['actor_optim'])
        print('[INFO] loaded the model and optimizer from', path)

    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        if not os.path.exists('./save'):
            os.mkdir('./save')

        params = {
                'actor_local': self.actor_local.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'actor_optim': self.actor_optimizer.state_dict(),
                'critic_local1': self.critic_local1.state_dict(),
                'critic_local2': self.critic_local2.state_dict(),
                'critic_target1': self.critic_target1.state_dict(),
                'critic_target2': self.critic_target2.state_dict(),
                'critic_optim1': self.critic_optimizer1.state_dict(),
                'critic_optim2': self.critic_optimizer2.state_dict()
                }

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        path = os.path.join('./save/td3_' + sha[:7] + '_ep_' +
                            str(n_episode)+'.pt')
        torch.save(params, path)
        print('[INFO] saved the model and optimizer to', path)

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch([self.actor_local, self.critic_local1,
                         self.critic_local2], log='parameters')

        step = 0
        for i_episode in range(1, hyper_params['EPISODE_NUM']+1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state, step)
                next_state, reward, done = self.step(state, action)

                if len(self.memory) >= hyper_params['BATCH_SIZE']:
                    experiences = self.memory.sample()
                    loss = self.update_model(experiences, step)
                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward
                step += 1

            else:
                if len(loss_episode) > 0:
                    avg_loss = np.array(loss_episode).mean()
                    print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                          % (i_episode, score, avg_loss))

                    if self.args.log:
                        wandb.log({'score': score, 'avg_loss': avg_loss})

                    if i_episode % self.args.save_period == 0:
                        self.save_params(i_episode)

        # termination
        self.env.close()

    def test(self):
        """Test the agent."""
        step = 0
        for i_episode in range(hyper_params['EPISODE_NUM']):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state, step)
                next_state, reward, done = self.step(state, action)

                state = next_state
                score += reward

            else:
                print('[INFO] episode %d\ttotal score: %d'
                      % (i_episode, score))

        # termination
        self.env.close()
