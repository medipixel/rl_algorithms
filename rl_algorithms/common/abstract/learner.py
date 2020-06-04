from abc import ABC, abstractmethod
import argparse
from typing import List, Tuple, Union

import torch
import torch.optim as optim

from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict

TensorTuple = Tuple[torch.Tensor, ...]


class Learner(ABC):
    """Abstract learner for all learners

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters

    """

    def __init__(self, args: argparse.Namespace, hyper_params: ConfigDict):
        self.args = args
        self.hyper_params = hyper_params

    @abstractmethod
    def update_model(
        self,
        networks: Tuple[Brain, ...],
        optimizer: Union[optim.Optimizer, Tuple[optim.Optimizer, ...]],
        experience: Union[TensorTuple, List[TensorTuple]],
    ) -> tuple:
        pass
