# -*- coding: utf-8 -*-
"""DPG agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://proceedings.mlr.press/v32/silver14.pdf
"""

import os
import git
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb

from baselines.dpg.model import Actor, Critic


# hyper parameters
hyper_params = {
        'GAMMA': 0.99,
        'MAX_EPISODE_STEPS': 500,
        'EPISODE_NUM': 1500
}


class Agent(object):
    """ActorCritic interacting with environment.

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        args (dict): arguments including hyperparameters and training settings
        device (torch.device): device selection (cpu / gpu)

    Attributes:
        env (gym.Env): openAI Gym environment with discrete action space
        actor (nn.Module): actor model to select actions
        critic (nn.Module): critic model to predict values
        args (dict): arguments including hyperparameters and training settings
        actor_optimizer (Optimizer): actor optimizer for training
        critic_optimizer (Optimizer): critic optimizer for training
        device (torch.device): device selection (cpu / gpu)

    """

    def __init__(self, env, args, device):
        """Initialization."""
        # environment setup
        self.env = env
        self.env._max_episode_steps = hyper_params['MAX_EPISODE_STEPS']

        # create a model
        self.device = device
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_low = float(self.env.action_space.low[0])
        action_high = float(self.env.action_space.high[0])
        self.actor = Actor(state_dim, action_dim, action_low,
                           action_high, self.device).to(self.device)
        self.critic = Critic(state_dim, action_dim,
                             self.device).to(self.device)

        # create optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

        # load stored parameters
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

    def update_model(self, experience):
        """Train the model after each episode."""
        state, action, reward, next_state, done = experience

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        value = self.critic(state, action)
        next_action = self.actor(next_state)
        next_value = self.critic(next_state, next_action).detach()
        curr_return = reward + (hyper_params['GAMMA'] * next_value * mask)
        curr_return = curr_return.to(self.device)

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

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(hyper_params)
            wandb.watch(self.actor, log='parameters')
            wandb.watch(self.critic, log='parameters')

        for i_episode in range(hyper_params['EPISODE_NUM']):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                loss = self.update_model((state, action, reward,
                                          next_state, done))

                state = next_state
                score += reward

                loss_episode.append(loss)  # for logging

            else:
                avg_loss = np.array(loss_episode).mean()
                print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                      % (i_episode+1, score, avg_loss))

                if self.args.log:
                    wandb.log({'score': score, 'avg_loss': avg_loss})

                if i_episode % self.args.save_period == 0:
                    self.save_params(i_episode)

        # termination
        self.env.close()

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
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            else:
                print('[INFO] episode %d\ttotal score: %d'
                      % (i_episode, score))

        # termination
        self.env.close()
