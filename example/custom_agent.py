# -*- coding: utf-8 -*-
"""Custom Agent for DQN.
   This is example for using custom agent.
   In this example, custom agent use state as exponential.
   You can customize any function e.g) select_aciton, train ... etc.

   To use custom agent just decorate class to build and import in main function.

    - Author: Jiseong Han
    - Contact: jisung.han@medipixel.io
"""

import numpy as np
import torch

from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.registry import AGENTS


@AGENTS.register_module
class CustomDQN(DQNAgent):
    """Example Custom Agent for DQN"""

    # pylint: disable=no-self-use
    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Preprocess state so that actor selects an action."""
        state = np.exp(state)
        state = numpy2floattensor(state, self.learner.device)
        return state

    def train(self):
        """Custom train."""
        pass
