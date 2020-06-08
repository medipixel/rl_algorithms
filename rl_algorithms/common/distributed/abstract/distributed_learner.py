from abc import abstractmethod
from typing import Tuple, Union

from rl_algorithms.common.abstract.learner import Learner, TensorTuple
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class DistributedLearnerWrapper(Learner):
    """Base wrapper class for distributed learners

    Attributes:
        learner (Learner): learner
        comm_config (ConfigDict): configs for communication

    """

    def __init__(self, learner: Learner, comm_config: ConfigDict):
        Learner.__init__(
            self, learner.args, learner.hyper_params, learner.log_cfg, learner.device
        )
        self.learner = learner
        self.comm_config = comm_config

    def _init_network(self):
        """Initialize learner networks and optimizers"""
        # no need to call this fn since the class takes initialized learner as input
        self.learner._init_network()

    @abstractmethod
    def _init_communication(self):
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

    def get_state_dicts(self) -> Union[Brain, Tuple[Brain]]:
        """Get the learner state_dicts that worker loads"""
        return self.learner.get_state_dicts()

    @abstractmethod
    def run(self):
        pass
