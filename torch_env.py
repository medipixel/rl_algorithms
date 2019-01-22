# -*- coding: utf-8 -*-
"""OpenAI gym wrapper to consistently use pytorch tensor.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

from typing import Tuple

import gym
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TorchEnv(gym.Env):
    """OpenAI gym wrapper to return pytorch tensors in step method.

    Args:
        env_name (str): open-ai environment name

    Attributes:
        env (gym.Env): open-ai gym instance
        state_dim (int): dimension of state space
        action_dim (int): dimension of action space
        action_low (float): lower bound of action value
        action_high (float): upper bound of action value

    """

    def __init__(self, env_name: str):
        """Initialization."""
        super(TorchEnv, self).__init__()

        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.observation_space.shape[0]
        self.action_low = float(self.env.action_space.low[0])
        self.action_high = float(self.env.action_space.high[0])

    def seed(self, seed: int):
        """Set seed."""
        self.env.seed(seed)

    def reset(self) -> torch.Tensor:
        """Reset the env."""
        init_state = self.env.reset()

        return torch.tensor(init_state).float().to(device)

    def render(self):
        """Reder the env."""
        self.env.render()

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step and return values in torch tensor type."""
        action = action.detach().to("cpu").numpy()
        next_state, reward, done, _ = self.env.step(action)

        next_state = torch.tensor(next_state).float().to(device)

        return (next_state, reward, done)

    def close(self):
        """Close the env."""
        self.env.close()

    def set_max_episode_steps(self, nstep: int):
        """Set max episode step number."""
        self.env._max_episode_steps = nstep
