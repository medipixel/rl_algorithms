from abc import ABC, abstractmethod
import argparse
from typing import Tuple, Union

import torch
import torch.optim as optim

from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class Learner(ABC):
    """Abstract learner for all learners"""

    def __init__(self, args: argparse.Namespace, hyper_params: ConfigDict):
        pass

    @abstractmethod
    def update_model(
        self,
        networks: Tuple[Brain, ...],
        optimizer: Union[optim.Optim, Tuple[optim.Optim, ...]],
        experience: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        pass
