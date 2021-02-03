"""DQN worker for distributed training.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import torch

from rl_algorithms.common.abstract.distributed_worker import DistributedWorker
from rl_algorithms.common.helper_functions import numpy2floattensor
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.registry import WORKERS, build_loss
from rl_algorithms.utils.config import ConfigDict


@WORKERS.register_module
class DQNWorker(DistributedWorker):
    """DQN worker for distributed training.

    Attributes:
        backbone (ConfigDict): backbone configs for building network
        head (ConfigDict): head configs for building network
        state_dict (ConfigDict): initial network state dict received form learner
        device (str): literal to indicate cpu/cuda use

    """

    def __init__(
        self,
        rank: int,
        device: str,
        hyper_params: ConfigDict,
        env_name: str,
        is_atari: bool,
        max_episode_steps: int,
        loss_type: ConfigDict,
        state_dict: OrderedDict,
        backbone: ConfigDict,
        head: ConfigDict,
        state_size: int,
        output_size: int,
    ):
        DistributedWorker.__init__(
            self, rank, device, hyper_params, env_name, is_atari, max_episode_steps
        )

        self.loss_fn = build_loss(loss_type)
        self.backbone_cfg = backbone
        self.head_cfg = head
        self.head_cfg.configs.state_size = state_size
        self.head_cfg.configs.output_size = output_size

        self.use_n_step = self.hyper_params.n_step > 1

        self.max_epsilon = self.hyper_params.max_epsilon
        self.min_epsilon = self.hyper_params.min_epsilon
        self.epsilon = self.hyper_params.max_epsilon

        self._init_networks(state_dict)

    # pylint: disable=attribute-defined-outside-init
    def _init_networks(self, state_dict: OrderedDict):
        """Initialize DQN policy with learner state dict."""
        self.dqn = Brain(self.backbone_cfg, self.head_cfg).to(self.device)
        self.dqn.load_state_dict(state_dict)
        self.dqn.eval()

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        DistributedWorker.load_params(self, path)

        params = torch.load(path)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        # epsilon greedy policy
        # pylint: disable=comparison-with-callable
        if self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state = self._preprocess_state(state, self.device)
                selected_action = self.dqn(state).argmax()
            selected_action = selected_action.cpu().numpy()

        # Decay epsilon
        self.epsilon = max(
            self.epsilon
            - (self.max_epsilon - self.min_epsilon) * self.hyper_params.epsilon_decay,
            self.min_epsilon,
        )

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def compute_priorities(self, memory: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute initial priority values of experiences in local memory."""
        states = numpy2floattensor(memory["states"], self.device)
        actions = numpy2floattensor(memory["actions"], self.device).long()
        rewards = numpy2floattensor(memory["rewards"].reshape(-1, 1), self.device)
        next_states = numpy2floattensor(memory["next_states"], self.device)
        dones = numpy2floattensor(memory["dones"].reshape(-1, 1), self.device)
        memory_tensors = (states, actions, rewards, next_states, dones)

        with torch.no_grad():
            dq_loss_element_wise, _ = self.loss_fn(
                self.dqn,
                self.dqn,
                memory_tensors,
                self.hyper_params.gamma,
                self.head_cfg,
            )
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params.per_eps
        return new_priorities

    def synchronize(self, new_state_dict: Dict[str, np.ndarray]):
        """Synchronize worker dqn with learner dqn."""
        self._synchronize(self.dqn, new_state_dict)
