# -*- coding: utf-8 -*-
"""PPO agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/abs/1707.06347
"""

import os
import git
import argparse
from collections import deque

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import util
from model import ActorCritic

import wandb  # 'wandb off' in the shell makes this diabled


# configurations
parser = argparse.ArgumentParser(description='TRPO with continuous\
                                             action example by Pytorch')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards')
parser.add_argument('--lambd', type=float, default=0.95,
                    help='discount factor for advantages')
parser.add_argument('--epsilon', type=float, default=0.2,
                    help='clipping parameter')
parser.add_argument('--entropy', type=float, default=1e-3,
                    help='entropy bonus')
parser.add_argument('--seed', type=int, default=777,
                    help='random seed for reproducibility')
parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2',
                    help='openai gym environment name\
                          (continuous action only)')
parser.add_argument('--epoch', type=int, default=4,
                    help='epoch size for each horizon')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the size of minibatch')
parser.add_argument('--max-episode-steps', type=int, default=300,
                    help='max steps per episode')
parser.add_argument('--horizon', type=int, default=512,
                    help='number of transitions to run training')
parser.add_argument('--episode-num', type=int, default=5000,
                    help='total episode number')
parser.add_argument('--model-path', type=str,
                    help='load the saved model and optimizer at the beginning')
parser.add_argument('--render-after', type=int, default=0,
                    help='start rendering after the input number of episode')
parser.add_argument('--no-render', dest='render', action='store_false',
                    help='turn off rendering')
parser.set_defaults(render=True)
parser.set_defaults(model_path=None)
args = parser.parse_args()

# initialization
env = gym.make(args.env)
env._max_episode_steps = args.max_episode_steps
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = float(env.action_space.low[0])
action_high = float(env.action_space.high[0])

# set random seed
env.seed(args.seed)
torch.manual_seed(args.seed)

# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent(object):
    """PPO Agent.

    Attributes:
        env (gym.Env): openAI Gym environment with discrete action space
        model (nn.Module): policy model + value estimator
        optimizer (Optimizer): optimizer for training actor-critic
        transition (list): list for storing a transition

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        model (nn.Module): policy model + value estimator

    """

    def __init__(self, env, model):
        """Initialization."""
        assert(issubclass(type(env), gym.Env))
        assert(issubclass(type(model), nn.Module))

        self.env = env
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.memory = deque()
        self.transition = []

        if args.model_path is not None and os.path.exists(args.model_path):
            self.load_params(args.model_path)

    def select_action(self, state):
        """Select an action from the input space."""
        selected_action, _, dist = self.model(state)
        self.transition += [state, dist.log_prob(selected_action).detach()]

        return selected_action

    def step(self, action):
        """Take an action and return the response of the env."""
        action = action.detach().to('cpu').numpy()
        next_state, reward, done, _ = self.env.step(action)

        self.transition += [action, reward, done]
        self.memory.append(self.transition)
        self.transition = []

        return next_state, reward, done

    def train(self):
        """Train the model after every N episodes."""
        states, log_probs, actions, rewards, dones = \
            util.decompose_memory(self.memory)
        losses = []
        for state, old_log_prob, action, reward, done in util.ppo_iter(
                                                          args.epoch,
                                                          args.batch_size,
                                                          states, log_probs,
                                                          actions, rewards,
                                                          dones):
            _, value, new_dist = self.model(state)
            entropy = args.entropy * new_dist.entropy().mean()

            # calculate returns and gae
            return_, advantage = util.get_ret_and_gae(reward, value, done,
                                                      args.gamma,
                                                      args.lambd)

            # normalize returns and  advantages
            return_ = (return_ - return_.mean()) /\
                      (return_.std() + 1e-7)
            advantage = (advantage - advantage.mean()) /\
                        (advantage.std() + 1e-7)

            # calculate ratios
            new_log_prob = new_dist.log_prob(action)
            ratio = (new_log_prob - old_log_prob).exp()

            # actor_loss
            surr_loss = ratio * advantage
            clipped_surr_loss = torch.clamp(ratio,
                                            1.0 - args.epsilon,
                                            1.0 + args.epsilon) * advantage
            actor_loss = -torch.min(surr_loss, clipped_surr_loss).mean()

            # critic_loss
            critic_loss = F.mse_loss(value, return_)

            # total_loss
            total_loss = critic_loss + actor_loss - args.entropy * entropy

            # backward and step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            losses.append(total_loss.data)

        return sum(losses) / len(losses)

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print('[ERROR] the input path does not exist. ->', path)
            return

        params = torch.load(path)
        self.model.load_state_dict(params['model_state_dict'])
        print('[INFO] loaded the model and optimizer from', path)

    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        if not os.path.exists('./save'):
            os.mkdir('./save')

        params = {'model_state_dict': self.model.state_dict()}

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        path = os.path.join('./save/ppo_' + sha[:7] + '_ep_' +
                            str(n_episode)+'.pt')
        torch.save(params, path)
        print('[INFO] saved the model and optimizer to', path)

    def run(self):
        """Run the agent."""
        # logger
        wandb.init()
        wandb.config.update(args)
        wandb.watch(self.model, log='parameters')

        scores = []
        for i_episode in range(args.episode_num):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if args.render and i_episode >= args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

                if len(self.memory) % args.horizon == 0:
                    loss = self.train()
                    self.memory.clear()

                    avg_score = sum(scores) / len(scores)
                    scores = []
                    print('[INFO] loss: %f, avg_score: %d' % (loss, avg_score))
                    wandb.log({'loss': loss, 'avg_score': avg_score})
            else:
                scores.append(score)
                print('[INFO] episode %d\ttotal score: %d'
                      % (i_episode+1, score))

        # termination
        self.env.close()
        self.save_params(args.episode_num)


if __name__ == '__main__':
    model = ActorCritic(state_dim, action_dim,
                        action_low, action_high).to(device)
    agent = Agent(env, model)
    agent.run()
