# -*- coding: utf-8 -*-
"""Actor-Critic agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import os
import git
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb

from baselines.ac.model import ActorCritic


# hyper parameters
hyper_params = {
        'GAMMA': 0.99,
        'STD': 1.0,
        'MAX_EPISODE_STEPS': 500,
        'EPISODE_NUM': 1500
}


class Agent(object):
    """ActorCritic interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment with discrete action space
        args (dict): arguments including hyperparameters and training settings
        device (str): device selection (cpu / gpu)

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        model (nn.Module): policy gradient model to select actions
        args (dict): arguments including hyperparameters and training settings
        optimizer (Optimizer): optimizer for training
        device (str): device selection (cpu / gpu)

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
        self.model = ActorCritic(hyper_params['STD'], state_dim,
                                 action_dim, action_low,
                                 action_high).to(self.device)

        # create optimizer
        self.optimizer = optim.Adam(self.model.parameters())

        # load stored parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        self.args = args

    def select_action(self, state):
        """Select an action from the input space."""
        state = torch.from_numpy(state).float().to(self.device)
        selected_action, predicted_value, dist = self.model(state)

        return (selected_action.detach().to('cpu').numpy(),
                dist.log_prob(selected_action).sum(),
                predicted_value)

    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done

    def update_model(self, done, log_prob, reward, next_state, curr_value):
        """Train the model after each episode."""
        next_state = torch.tensor(next_state).float().to(self.device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        if not done:
            next_value = self.model.critic(next_state).detach()
            curr_return = reward + hyper_params['GAMMA'] * next_value
        else:
            curr_return = torch.tensor(reward)

        curr_return = curr_return.float().to(self.device)

        # delta = G_t - v(s_t)
        delta = curr_return - curr_value.detach()

        # calculate loss at the current step
        policy_loss = -delta * log_prob  # delta is not backpropagated
        value_loss = F.mse_loss(curr_value, curr_return)
        loss = policy_loss + value_loss

        # train
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print('[ERROR] the input path does not exist. ->', path)
            return

        params = torch.load(path)
        self.model.load_state_dict(params['model_state_dict'])
        self.optimizer.load_state_dict(params['optim_state_dict'])
        print('[INFO] loaded the model and optimizer from', path)

    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        if not os.path.exists('./save'):
            os.mkdir('./save')

        params = {
                 'model_state_dict': self.model.state_dict(),
                 'optim_state_dict': self.optimizer.state_dict()
                 }

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        path = os.path.join('./save/actor_critic_' +
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
            wandb.watch(self.model, log='parameters')

        for i_episode in range(1, hyper_params['EPISODE_NUM']+1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action, log_prob, predicted_value = self.select_action(state)
                next_state, reward, done = self.step(action)
                loss = self.update_model(done, log_prob, reward,
                                         next_state, predicted_value)
                loss_episode.append(loss)

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

                action, log_prob, predicted_value = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            else:
                print('[INFO] episode %d\ttotal score: %d'
                      % (i_episode, score))

        # termination
        self.env.close()
