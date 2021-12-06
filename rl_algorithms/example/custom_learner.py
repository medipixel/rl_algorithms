"""This is example to use custom learner that inherit DQNLearner.
    You need to decorate class to register your own Learner to build.
    And import custom learner on main file.

    If you want to make custom learner, you can inherit BaseLeaner or Learner.
    If you make your own learner, you need to change config file to build.

    - Author: Jiseong Han
    - Contact: jisung.han@medipixel.io
"""
from typing import Tuple, Union

import numpy as np
import torch

from rl_algorithms.common.abstract.learner import TensorTuple
from rl_algorithms.dqn.learner import DQNLearner
from rl_algorithms.registry import LEARNERS


@LEARNERS.register_module
class CustomDQNLearner(DQNLearner):
    """Example of Custom DQN learner."""

    def _init_network(self):
        return super()._init_network()

    def update_model(
        self, experience: Union[TensorTuple, Tuple[TensorTuple]]
    ) -> Tuple[torch.Tensor, torch.Tensor, list, np.ndarray]:  # type: ignore
        """
        Custom Update model with experience.
        """
        pass
