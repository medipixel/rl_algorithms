"""Worker classes for distributed training

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

from abc import ABC, abstractmethod
import argparse
import os
import random
from typing import Deque, Dict, List, Tuple

import gym
import numpy as np
import torch

from rl_algorithms.common.env.atari_wrappers import atari_env_generator
import rl_algorithms.common.env.utils as env_utils
from rl_algorithms.common.helper_functions import set_random_seed
from rl_algorithms.utils.config import ConfigDict


class BaseWorker(ABC):
    """Base class for Worker classes."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        pass

    @abstractmethod
    def synchronize(self, new_params: list):
        pass

    # pylint: disable=no-self-use
    def _synchronize(self, network, new_params: List[np.ndarray]):
        """Copy parameters from numpy arrays."""
        for param, new_param in zip(network.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)


class Worker(BaseWorker):
    """Base class for all functioning RL workers.

    Attributes:
        rank (int): rank (ID) of worker
        args (argparse.Namespace): args from run script
        env_info (ConfigDict): information about environment
        hyper_params (ConfigDict): algorithm hyperparameters
        device (torch.Device): device on which worker process runs
        env (gym.ENV): gym environment
    """

    def __init__(
        self,
        rank: int,
        args: argparse.Namespace,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        device: str,
    ):
        """Initialize."""
        self.rank = rank
        self.args = args
        self.env_info = env_info
        self.hyper_params = hyper_params
        self.device = torch.device(device)

        self._init_env()

    # pylint: disable=attribute-defined-outside-init, no-self-use
    def _init_env(self):
        """Intialize worker local environment."""
        if self.env_info.is_atari:
            self.env = atari_env_generator(
                self.env_info.name, self.args.max_episode_steps, frame_stack=True
            )
        else:
            self.env = gym.make(self.env_info.name)
            env_utils.set_env(self.env, self.args)

        random.seed(self.rank)
        env_seed = random.randint(0, 999)
        set_random_seed(env_seed, self.env)

    @abstractmethod
    def load_params(self, path: str):
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        pass

    # NOTE: No need to explicitly implement for non-PER/non-Ape-X workers
    @abstractmethod
    def compute_priorities(self, experience: Dict[str, np.ndarray]):
        pass

    @abstractmethod
    def synchronize(self, new_params: list):
        pass

    # pylint: disable=no-self-use
    @staticmethod
    def _preprocess_state(state: np.ndarray, device: torch.device) -> torch.Tensor:
        state = torch.FloatTensor(state).to(device)
        return state


class DistributedWorkerWrapper(BaseWorker):
    """Base wrapper class for distributed worker wrappers."""

    def __init__(self, worker: Worker, args: argparse.Namespace, comm_cfg: ConfigDict):
        self.worker = worker
        self.args = args
        self.comm_cfg = comm_cfg

    @abstractmethod
    def init_communication(self):
        pass

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        return self.worker.select_action(state)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        return self.worker.step(action)

    def synchronize(self, new_params: list):
        """Synchronize worker brain with learner brain."""
        self.worker.synchronize(new_params)

    @abstractmethod
    def collect_data(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def run(self):
        pass

    def preprocess_nstep(self, nstepqueue: Deque) -> Tuple[np.ndarray, ...]:
        """Return n-step transition with discounted reward."""
        discounted_reward = 0
        _, _, _, last_state, done = nstepqueue[-1]
        for transition in list(reversed(nstepqueue)):
            state, action, reward, _, _ = transition
            discounted_reward = reward + self.hyper_params.gamma * discounted_reward
        nstep_data = (state, action, discounted_reward, last_state, done)

        return nstep_data
