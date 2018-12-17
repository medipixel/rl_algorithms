# -*- coding: utf-8 -*-
"""Deep Deterministic Policy Gradient Algorithm.

This module demonstrates DDPG model on the environment
with continuous action space in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1509.02971.pdf
"""

import argparse
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpg_agent import DDPGAgent


# configurations
parser = argparse.ArgumentParser(description='DDPG with continuous\
                                             action example by Pytorch')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
parser.add_argument('--tau', type=float, default=1e-3,
                    help='soft-update rate')
parser.add_argument('--seed', type=int, default=777,
                    help='random seed for reproducibility')
parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2',
                    help='openai gym environment name\
                          (continuous action only)')
parser.add_argument('--buffer-size', type=int, default=10000,
                    help='replay memory size')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size')
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


class DDPGActor(nn.Module):
    """DDPG actor model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        actor (nn.Sequential): actor model with FC layers

    """

    def __init__(self, state_dim, action_dim):
        """Initialization."""
        super(DDPGActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = nn.Sequential(
                        nn.Linear(self.state_dim, 24),
                        nn.ReLU(),
                        nn.Linear(24, 48),
                        nn.ReLU(),
                        nn.Linear(48, 24),
                        nn.ReLU(),
                        nn.Linear(24, self.action_dim),
                        nn.Tanh()
                     )

    def forward(self, state):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            specific action

        """
        state = torch.tensor(state).float().to(device)
        action = self.actor(state)

        # adjust the output range to [action_low, action_high]
        scale_factor = (action_high - action_low) / 2
        reloc_factor = (action_high - scale_factor)
        action = action * scale_factor + reloc_factor
        action = torch.clamp(action, action_low, action_high)

        return action


class DDPGCritic(nn.Module):
    """DDPG critic model with simple FC layers.

    Args:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space

    Attributes:
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        critic (nn.Sequential): critic model with FC layers

    """

    def __init__(self, state_dim, action_dim):
        """Initialization."""
        super(DDPGCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.state_dim+self.action_dim, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, 1)

    def forward(self, state, action):
        """Forward method implementation.

        Args:
            state (numpy.ndarray): input vector on the state space

        Returns:
            predicted state value

        """
        state = torch.tensor(state).float().to(device)

        x = torch.cat((state, action), dim=-1)  # concat action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        predicted_value = self.fc4(x)

        return predicted_value


if __name__ == '__main__':
    actor_local = DDPGActor(state_dim, action_dim).to(device)
    actor_target = DDPGActor(state_dim, action_dim).to(device)
    actor_target.load_state_dict(actor_local.state_dict())

    critic_local = DDPGCritic(state_dim, action_dim).to(device)
    critic_target = DDPGCritic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic_local.state_dict())

    agent = DDPGAgent(env, actor_local, actor_target,
                      critic_local, critic_target, args)
    agent.run()
