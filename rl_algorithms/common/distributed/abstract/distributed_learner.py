"""Base wrapper class for distributed learners

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

from abc import abstractmethod
from typing import Tuple, Union

from rl_algorithms.common.abstract.learner import Learner, LearnerWrapper, TensorTuple
from rl_algorithms.utils.config import ConfigDict


class DistributedLearnerWrapper(LearnerWrapper):
    """Base wrapper class for distributed learners

    Attributes:
        learner (Learner): learner
        comm_config (ConfigDict): configs for communication

    """

    def __init__(self, learner: Learner, comm_cfg: ConfigDict):
        LearnerWrapper.__init__(self, learner)
        self.comm_cfg = comm_cfg

    @abstractmethod
    def init_communication(self):
        pass

    def update_model(self, experience: Union[TensorTuple, Tuple[TensorTuple]]) -> tuple:
        """Run one step of learner model update"""
        return self.learner.update_model(experience)

    def save_params(self, n_update_step: int):
        """Save learner params at defined directory"""
        self.learner.save_params(n_update_step)

    def load_params(self, path: str):
        """Load params at start"""
        self.learner.load_params(path)

    def get_policy(self):
        return self.learner.get_policy()

    def get_state_dict(self):
        return self.learner.get_state_dict()

    @abstractmethod
    def run(self):
        pass
