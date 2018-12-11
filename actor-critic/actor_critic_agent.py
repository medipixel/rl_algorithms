# -*- coding: utf-8 -*-
"""Actor-Critic agent for episodic tasks in OpenAI Gym.
- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import os
import git
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import wandb  # 'wandb off' in the shell makes this diabled


# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ActorCriticAgent(object):
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

    def select_action(self, state):
        """Select an action from the input space."""
        state = torch.from_numpy(state).float().to(device)
        selected_action, predicted_value, dist = self.model(state)

        return (selected_action.detach().to('cpu').numpy(),
                dist.log_prob(selected_action).sum(),
                predicted_value)

    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action)

        return next_state, reward, done

    def train(self, done, log_prob, reward, next_state, curr_value):
        """Train the model after each episode."""
        next_state = torch.tensor(next_state).float().to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        if not done:
            next_value = self.model.critic(next_state).detach()
            curr_return = reward + self.args.gamma * next_value
        else:
            curr_return = torch.tensor(reward)

        curr_return = curr_return.float().to(device)

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
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action, log_prob, predicted_value = self.select_action(state)
                next_state, reward, done = self.step(action)
                loss = self.train(done, log_prob, reward,
                                  next_state, predicted_value)
                loss_episode.append(loss)

                state = next_state
                score += reward

            else:
                avg_loss = np.array(loss_episode).mean()
                print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                      % (i_episode, score, avg_loss))
#                wandb.log({'score': score, 'avg_loss': avg_loss})

        # termination
        self.env.close()
        self.save_params(self.args.episode_num)
