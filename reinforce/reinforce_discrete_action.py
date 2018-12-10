# -*- coding: utf-8 -*-
"""Reinforce method for the environments with discrete action in OpenAI Gym.

This module demonstrates Reinforce with baseline model on the environment
with discrete action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from reinforce_agent import ReinforceAgent


# configurations
parser = argparse.ArgumentParser(description='Reinforce with discrete action\
                                             example by Pytorch')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
parser.add_argument('--seed', type=int, default=777,
                    help='random seed for reproducibility')
parser.add_argument('--env', type=str, default='CartPole-v0',
                    help='openai gym environment name (discrete action only)')
parser.add_argument('--max-episode-steps', type=int, default=500,
                    help='max steps per episode')
parser.add_argument('--episode-num', type=int, default=1000,
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
action_dim = env.action_space.n

# set random seed
env.seed(args.seed)
torch.manual_seed(args.seed)


class ReinforceDiscreteAction(nn.Module):
    """Reinforce discrete action model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        actor (nn.Sequential): actor model with FC layers
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim, action_dim):
        """Initialization."""
        super(ReinforceDiscreteAction, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = nn.Sequential(
                        nn.Linear(self.state_dim, 24),
                        nn.ReLU(),
                        nn.Linear(24, 48),
                        nn.ReLU(),
                        nn.Linear(48, 24),
                        nn.ReLU(),
                        nn.Linear(24, self.action_dim)
                     )

        self.critic = nn.Sequential(
                        nn.Linear(self.state_dim, 24),
                        nn.ReLU(),
                        nn.Linear(24, 48),
                        nn.ReLU(),
                        nn.Linear(48, 24),
                        nn.ReLU(),
                        nn.Linear(24, 1)
                )

    def forward(self, state):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            softmax distribution as the output of actor model and
            approximated value of the input state as the output of
            critic model.

        """
        action_probs = F.softmax(self.actor(state), dim=-1)
        predicted_value = self.critic(state)
        dist = Categorical(action_probs)
        selected_action = dist.sample()

        return selected_action, predicted_value, dist


if __name__ == '__main__':
    model = ReinforceDiscreteAction(state_dim, action_dim)
    agent = ReinforceAgent(env, model, args)
    agent.run()
