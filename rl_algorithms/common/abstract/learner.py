from abc import ABC, abstractmethod
import argparse
import os
from typing import Tuple, Union

import torch

from rl_algorithms.utils.config import ConfigDict

TensorTuple = Tuple[torch.Tensor, ...]


class Learner(ABC):
    """Abstract learner for all learners

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log and checkpoint

    """

    def __init__(
        self,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        log_cfg: ConfigDict,
        device: torch.device,
    ):
        self.args = args
        self.hyper_params = hyper_params
        self.log_cfg = log_cfg
        self.device = device

    @abstractmethod
    def _init_network(self):
        pass

    @abstractmethod
    def update_model(self, experience: Union[TensorTuple, Tuple[TensorTuple]]) -> tuple:
        pass

    @abstractmethod
    def save_params(self, n_episode: int):
        pass

    def _save_params(self, params: dict, n_episode: int):
        """Save parameters of networks."""
        os.makedirs(self.log_cfg["ckpt_path"], exist_ok=True)

        path = os.path.join(
            self.log_cfg["ckpt_path"]
            + self.log_cfg["sha"]
            + "_ep_"
            + str(n_episode)
            + ".pt"
        )
        torch.save(params, path)

        print("[INFO] Saved the model and optimizer to", path)

    @abstractmethod
    def load_params(self, path: str):
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )
