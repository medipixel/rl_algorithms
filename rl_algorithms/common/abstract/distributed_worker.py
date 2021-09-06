"""Worker classes for distributed training.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

from abc import ABC, abstractmethod
import os
import random
from typing import Deque, Dict, Tuple

import gym
import numpy as np
import torch

from rl_algorithms.common.env.atari_wrappers import atari_env_generator
import rl_algorithms.common.env.utils as env_utils
from rl_algorithms.common.helper_functions import numpy2floattensor, set_random_seed
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class BaseDistributedWorker(ABC):
    """Base class for Worker classes."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        pass

    @abstractmethod
    def synchronize(self, new_state_dict: Dict[str, np.ndarray]):
        pass

    # pylint: disable=no-self-use
    def _synchronize(self, network: Brain, new_state_dict: Dict[str, np.ndarray]):
        """Copy parameters from numpy arrays."""
        param_name_list = list(new_state_dict.keys())
        for worker_named_param in network.named_parameters():
            worker_param_name = worker_named_param[0]
            if worker_param_name in param_name_list:
                new_param = numpy2floattensor(
                    new_state_dict[worker_param_name], self.device
                )
                worker_named_param[1].data.copy_(new_param)


class DistributedWorker(BaseDistributedWorker):
    """Base class for all functioning RL workers.

    Attributes:
        rank (int): rank (ID) of worker
        hyper_params (ConfigDict): algorithm hyperparameters
        device (torch.Device): device on which worker process runs
        env (gym.ENV): gym environment
    """

    def __init__(
        self,
        rank: int,
        device: str,
        hyper_params: ConfigDict,
        env_name: str,
        is_atari: bool,
        max_episode_steps: int,
    ):
        """Initialize."""
        self.rank = rank
        self.device = torch.device(device)

        self.hyper_params = hyper_params
        self.env_name = env_name
        self.is_atari = is_atari
        self.max_episode_steps = max_episode_steps

        self._init_env()

    # pylint: disable=attribute-defined-outside-init, no-self-use
    def _init_env(self):
        """Intialize worker local environment."""
        if self.is_atari:
            self.env = atari_env_generator(
                self.env_name, self.max_episode_steps, frame_stack=True
            )
        else:
            self.env = gym.make(self.env_name)
            env_utils.set_env(self.env, self.max_episode_steps)

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
    def synchronize(self, new_state_dict: Dict[str, np.ndarray]):
        pass

    @staticmethod
    def _preprocess_state(state: np.ndarray, device: torch.device) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        state = numpy2floattensor(state, device)
        return state


class DistributedWorkerWrapper(BaseDistributedWorker):
    """Base wrapper class for distributed worker wrappers."""

    def __init__(self, worker: DistributedWorker, comm_cfg: ConfigDict):
        self.worker = worker
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

    def synchronize(self, new_state_dict: Dict[str, np.ndarray]):
        """Synchronize worker brain with learner brain."""
        self.worker.synchronize(new_state_dict)

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
