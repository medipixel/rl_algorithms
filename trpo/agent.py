# -*- coding: utf-8 -*-
"""TRPO agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: http://arxiv.org/abs/1502.05477
"""

import os
import git
import argparse
from collections import deque

import gym
import torch
import torch.nn as nn
import scipy.optimize

import utils
from models import Actor, Critic

# import wandb  # 'wandb off' in the shell makes this diabled


# configurations
parser = argparse.ArgumentParser(description='TRPO with continuous\
                                             action example by Pytorch')
parser.add_argument('--gamma', type=float, default=0.98,
                    help='discount factor for rewards')
parser.add_argument('--lambd', type=float, default=0.92,
                    help='discount factor for advantages')
parser.add_argument('--max-kl', type=float, default=1e-2,
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1,
                    help='damping (default: 1e-1)')
parser.add_argument('--l2-reg', type=float, default=1e-3,
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--seed', type=int, default=777,
                    help='random seed for reproducibility')
parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2',
                    help='openai gym environment name\
                          (continuous action only)')
parser.add_argument('--episodes-per-batch', type=int, default=20,
                    help='the number of episodes per batch')
parser.add_argument('--max-episode-steps', type=int, default=300,
                    help='max steps per episode')
parser.add_argument('--episode-num', type=int, default=2000,
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
    """TRPO Agent.

    Attributes:
        env (gym.Env): openAI Gym environment with discrete action space
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to predict values

    Args:
        env (gym.Env): openAI Gym environment with discrete action space
        actor (nn.Module): policy gradient model to select actions
        critic (nn.Module): policy gradient model to select actions

    """

    def __init__(self, env, actor, critic):
        """Initialization."""
        assert(issubclass(type(env), gym.Env))
        assert(issubclass(type(actor), nn.Module))
        assert(issubclass(type(critic), nn.Module))

        self.env = env
        self.actor = actor
        self.critic = critic
        if args.model_path is not None and os.path.exists(args.model_path):
            self.load_params(args.model_path)
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

        # calculate returns and gae
        returns, advantages = utils.get_ret_and_gae(rewards, values, dones,
                                                    args.gamma,
                                                    args.lambd)

        # normalize the advantages
        advantages = (advantages - advantages.mean()) /\
                     (advantages.std() + 1e-7)

        # train actor
        actor_loss = utils.trpo_step(self.actor, states, actions, advantages,
                                     args.max_kl, args.damping)

        # train critic
        targets = returns.detach()
        get_value_loss = utils.ValueLoss(self.critic, states,
                                         targets, args.l2_reg)
        flat_params, _, _ = \
            scipy.optimize.fmin_l_bfgs_b(
                    get_value_loss,
                    utils.get_flat_params_from(
                     self.critic).to('cpu').double().numpy(),
                    maxiter=25)
        utils.set_flat_params_to(self.critic, torch.Tensor(flat_params))
        critic_loss = (self.critic(states) - targets).pow(2).mean()

        # for logging
        total_loss = actor_loss + critic_loss

        return total_loss.data

    def load_params(self, path):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print('[ERROR] the input path does not exist. ->', path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params['actor_state_dict'])
        self.critic.load_state_dict(params['critic_state_dict'])
        print('[INFO] loaded the model and optimizer from', path)

    def save_params(self, n_episode):
        """Save model and optimizer parameters."""
        if not os.path.exists('./save'):
            os.mkdir('./save')

        params = {'actor_state_dict': self.actor.state_dict(),
                  'critic_state_dict': self.critic.state_dict()}

        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        path = os.path.join('./save/trpo_' + sha[:7] + '_ep_' +
                            str(n_episode)+'.pt')
        torch.save(params, path)
        print('[INFO] saved the model and optimizer to', path)

    def run(self):
        """Run the agent."""
        # logger
#         wandb.init()
#         wandb.config.update(args)
#         wandb.watch(self.actor, log='parameters')
#         wandb.watch(self.critic, log='parameters')

        scores = []
        for i_episode in range(args.episode_num):
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if args.render and i_episode >= args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(state, action)

                state = next_state
                score += reward
            else:
                scores.append(score)

            # train every self.args.epicodes_per_batch
            if i_episode % args.episodes_per_batch == 0:
                loss = self.train()
                avg_score = sum(scores) / len(scores)
                scores.clear()

                print('[INFO] episode %d\ttotal score: %d\tloss: %f'
                      % (i_episode, avg_score, loss))
#                wandb.log({'score': avg_score, 'loss': loss})

        # termination
        self.env.close()
        self.save_params(args.episode_num)


if __name__ == '__main__':
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    agent = Agent(env, actor, critic)
    agent.run()
