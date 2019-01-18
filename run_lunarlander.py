# -*- coding: utf-8 -*-
"""Training or testing baselines on LunarLander-v2 or LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import torch
import gym


# configurations
parser = argparse.ArgumentParser(description='Pytorch RL baselines')
parser.add_argument('--seed', type=int, default=777,
                    help='random seed for reproducibility')
parser.add_argument('--algo', type=str, default='trpo',
                    help='choose an algorithm')
parser.add_argument('--test', dest='test', action='store_true',
                    help='test mode (no training)')
parser.add_argument('--load-from', type=str,
                    help='load the saved model and optimizer at the beginning')
parser.add_argument('--off-render', dest='render', action='store_false',
                    help='turn off rendering')
parser.add_argument('--render-after', type=int, default=0,
                    help='start rendering after the input number of episode')
parser.add_argument('--log', dest='log', action='store_true',
                    help='turn on logging')
parser.add_argument('--save-period', type=int, default=50,
                    help='save model period')
parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=True)
parser.set_defaults(log=False)
args = parser.parse_args()

# device selection: cpu / gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# algorithms
policy_gradients = {'ac', 'reinforce', 'dpg', 'ddpg', 'trpo', 'ppo', 'bc'}

# import the agent
if args.algo == 'reinforce':
    from baselines.reinforce.agent import Agent
elif args.algo == 'ac':
    from baselines.ac.agent import Agent
elif args.algo == 'dpg':
    from baselines.dpg.agent import Agent
elif args.algo == 'ddpg':
    from baselines.ddpg.agent import Agent
elif args.algo == 'trpo':
    from baselines.trpo.agent import Agent
elif args.algo == 'ppo':
    from baselines.ppo.agent import Agent
elif args.algo == 'bc':
    from baselines.bc.agent import Agent


def main():
    """Main."""
    # choose an env
    if args.algo in policy_gradients:
        env = 'LunarLanderContinuous-v2'

    # env initialization
    env = gym.make(env)

    # set a random seed
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # create an agent
    agent = Agent(env, args, device)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()


if __name__ == '__main__':
    main()
