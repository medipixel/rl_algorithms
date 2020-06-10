from abc import ABC, abstractmethod
import argparse
from collections import OrderedDict
import os
import shutil
import subprocess
from typing import Tuple, Union

import torch

from rl_algorithms.utils.config import ConfigDict

TensorTuple = Tuple[torch.Tensor, ...]


class Learner(ABC):
    """Abstract class for all learner objects."""

    @abstractmethod
    def update_model(self, experience: Union[TensorTuple, Tuple[TensorTuple]]) -> tuple:
        pass

    @abstractmethod
    def save_params(self, n_episode: int):
        pass

    @abstractmethod
    def load_params(self, path: str):
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )

    @abstractmethod
    def get_state_dict(self) -> Union[OrderedDict, Tuple[OrderedDict]]:
        pass


class BaseLearner(Learner):
    """Base class for all base learners.

    Attributes:
        args (argparse.Namespace): arguments including hyperparameters and training settings
        hyper_params (ConfigDict): hyper-parameters
        log_cfg (ConfigDict): configuration for saving log
        sha (str): sha code of current git commit

    """

    def __init__(
        self,
        args: argparse.Namespace,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        log_cfg: ConfigDict,
        device: torch.device,
    ):
        """Initialize."""
        self.args = args
        self.env_info = env_info
        self.hyper_params = hyper_params
        self.device = device

        if not self.args.test:
            self.ckpt_path = (
                "./checkpoint/"
                f"{env_info.env_name}/{log_cfg.agent}/{log_cfg.curr_time}/"
            )
            os.makedirs(self.ckpt_path, exist_ok=True)

            # save configuration
            shutil.copy(self.args.cfg_path, os.path.join(self.ckpt_path, "config.py"))

        # for logging
        self.sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])[:-1]
            .decode("ascii")
            .strip()
        )

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
        os.makedirs(self.ckpt_path, exist_ok=True)

        path = os.path.join(self.ckpt_path + self.sha + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)

        print("[INFO] Saved the model and optimizer to", path)

    @abstractmethod
    def load_params(self, path: str):
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )

    @abstractmethod
    def get_state_dict(self) -> Union[OrderedDict, Tuple[OrderedDict]]:
        pass


class LearnerWrapper(Learner):
    """Base class for all learner wrappers"""

    def __init__(self, learner: BaseLearner):
        """Initialize."""
        self.learner = learner

    def update_model(self, experience: Union[TensorTuple, Tuple[TensorTuple]]) -> tuple:
        return self.learner.update_model(experience)

    def save_params(self, n_episode: int):
        self.learner.save_params(n_episode)

    def load_params(self, path: str):
        self.learner.load_params(path)

    def get_state_dict(self) -> Union[OrderedDict, Tuple[OrderedDict]]:
        return self.learner.get_state_dict()
