# -*- coding: utf-8 -*-
"""ActorCritic method for episodic tasks with continuous actions in OpenAI Gym.

This module demonstrates Actor-Critic with baseline model on the episodic tasks
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import gym

import torch
import torch.nn as nn
from torch.distributions import Normal

from actor_critic_agent import ActorCriticAgent


# configurations
parser = argparse.ArgumentParser(description='Actor-Critic with continuous action\
                                             example by Pytorch')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
parser.add_argument('--std', type=float, default=1.0,
                    help='standard deviation for normal distribution')
parser.add_argument('--seed', type=int, default=777,
                    help='random seed for reproducibility')
parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2',
                    help='openai gym environment name\
                          (continuous action only)')
parser.add_argument('--max-episode-steps', type=int, default=500,
                    help='max steps per episode')
parser.add_argument('--episode-num', type=int, default=1500,
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


class ActorCriticContinuousAction(nn.Module):
    """Actor-Critic continuous action model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        actor_mu (nn.Sequential): actor model for mu with FC layers
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim, action_dim):
        """Initialization."""
        super(ActorCriticContinuousAction, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_mu = nn.Sequential(
                            nn.Linear(self.state_dim, 24),
                            nn.ReLU(),
                            nn.Linear(24, 48),
                            nn.ReLU(),
                            nn.Linear(48, 24),
                            nn.ReLU(),
                            nn.Linear(24, self.action_dim),
                            nn.Tanh()
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

        The original paper suggests to employ an approximator
        for standard deviation, but, practically, it shows worse performance
        rather than using constant value by setting a hyper-parameter.
        The default std is 1.0 that leads to a good result on
        LunarLanderContinuous-v2 environment.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            normal distribution parameters as the output of actor model and
            approximated value of the input state as the output of
            critic model

        """
        norm_dist_mu = self.actor_mu(state)
        norm_dist_std = args.std
        predicted_value = self.critic(state)

        dist = Normal(norm_dist_mu, norm_dist_std)
        selected_action = torch.clamp(dist.rsample(), action_low, action_high)

        return selected_action, predicted_value, dist


if __name__ == '__main__':
    model = ActorCriticContinuousAction(state_dim, action_dim)
    agent = ActorCriticAgent(env, model, args)
    agent.run()
