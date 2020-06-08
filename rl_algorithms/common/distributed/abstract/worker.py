from abc import ABC, abstractmethod
import argparse
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import torch

from rl_algorithms.utils.config import ConfigDict


class Worker(ABC):
    """Abstract class for distributed workers

    Attributes
        rank (int): rank of worker process
        args (argparse.Namespace): args
        comm_config (ConfigDict): config of inter-process communication
        param_queue (deque): queue to store received new network parameters
        device (torch.device): device for worker

    """

    def __init__(
        self, rank: int, args: argparse.Namespace, comm_cfg: ConfigDict,
    ):
        """Initialize"""
        self.rank = rank
        self.args = args
        self.comm_cfg = comm_cfg
        self.param_queue = deque(maxlen=5)
        self.device = torch.device(self.args["worker_device"])

    @abstractmethod
    def _init_networks(self):
        pass

    @abstractmethod
    def _init_communication(self):
        pass

    # pylint: disable=no-self-use
    @staticmethod
    def _preprocess_state(state: np.ndarray, device: torch.device) -> torch.Tensor:
        state = torch.FloatTensor(state).to(device)
        return state

    @abstractmethod
    def select_action(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        pass

    @abstractmethod
    def collect_data(self) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def synchronize(self, new_params: list):
        pass

    # pylint: disable=no-self-use
    def _synchronize(self, network, new_params: List[np.ndarray]):
        """Copy parameters from numpy arrays"""
        for param, new_param in zip(network.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)

    @abstractmethod
    def run(self):
        pass
