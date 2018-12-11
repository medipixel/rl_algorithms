# -*- coding: utf-8 -*-
"""Reinforce agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import os
import git
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import wandb  # 'wandb off' in the shell makes this diabled


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ReinforceAgent(object):
    """ReinforceAgent interacting with environment.

    Attributes:
        env (gym.Env): openAI Gym environment with discrete action space
        model (nn.Module): policy gradient model to select actions
        args (dict): arguments including hyperparameters and training settings

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        model (nn.Module): policy gradient model to select actions
        args (dict): arguments including hyperparameters and training settings
        optimizer (Optimizer): optimizer for training
        log_prob_sequence (list): log probabailities of an episode
        predicted_value_sequence (list): predicted values of an episode
        reward_sequence (list): rewards of an episode to calculate returns

    """

    def __init__(self, env, model, args):
        """Initialization."""
        assert(issubclass(type(env), gym.Env))
        assert(issubclass(type(model), nn.Module))

        self.env = env
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters())
        if args.model_path is not None and os.path.exists(args.model_path):
            self.load_params(args.model_path)
        self.args = args

        self.log_prob_sequence = []
        self.predicted_value_sequence = []
        self.reward_sequence = []

    def select_action(self, state):
        """Select an action from the input space."""
        state = torch.from_numpy(state).float().to(device)
        selected_action, predicted_value, dist = self.model(state)

        self.log_prob_sequence.append(dist.log_prob(selected_action).sum())
        self.predicted_value_sequence.append(predicted_value)

        return selected_action.detach().to('cpu').numpy()

    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        # store rewards to calculate return values
        self.reward_sequence.append(reward)

        return next_state, reward, done

    def train(self):
        """Train the model after each episode."""
        return_value = 0  # initial return value
        return_sequence = deque()

        # calculate return value at each step
        for i in range(len(self.reward_sequence)-1, -1, -1):
            return_value = self.reward_sequence[i] +\
                           self.args.gamma *\
                           return_value
            return_sequence.appendleft(return_value)

        return_sequence = torch.tensor(return_sequence).to(device)
        # standardize returns for better stability
        return_sequence =\
            (return_sequence - return_sequence.mean()) /\
            (return_sequence.std() + 1e-7)

        # calculate loss at each step
        loss_sequence = []
        for log_prob,\
            return_value,\
            predicted_value in zip(self.log_prob_sequence,
                                   return_sequence,
                                   self.predicted_value_sequence):
            delta = return_value - predicted_value.detach()

            policy_loss = -delta*log_prob
            value_loss = F.smooth_l1_loss(predicted_value, return_value)

            loss = (policy_loss + value_loss)
            loss_sequence.append(loss)

        # train
        self.optimizer.zero_grad()
        total_loss = torch.stack(loss_sequence).sum()
        total_loss.backward()
        self.optimizer.step()

        # clear
        self.log_prob_sequence.clear()
        self.predicted_value_sequence.clear()
        self.reward_sequence.clear()

        return total_loss.data

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
        path = os.path.join('./save/reinforce_' +
                            sha[:7] +
                            '_ep_' +
                            str(n_episode)+'.pt')
        torch.save(params, path)
        print('[INFO] saved the model and optimizer to', path)

    def run(self):
        """Run the agent."""
        # logger
#        wandb.init()
#        wandb.config.update(self.args)
#        wandb.watch(self.model, log='parameters')

        for i_episode in range(self.args.episode_num):
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
                loss = self.train()
                print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                      % (i_episode, score, loss))
#                wandb.log({'score': score, 'loss': loss})

        # termination
        self.env.close()
        self.save_params(self.args.episode_num)
