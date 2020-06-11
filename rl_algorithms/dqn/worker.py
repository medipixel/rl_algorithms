import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import ray
import torch

from rl_algorithms.common.distributed.apex.worker import ApeXWorker
from rl_algorithms.registry import build_loss
from rl_algorithms.utils.config import ConfigDict


@ray.remote
class ApeXDQNWorker(ApeXWorker):
    """ApeX DQN Worker Class

    Attributes:
        loss_fn (LOSS): loss function for computing priority value
        dqn (Brain): worker brain

    """

    def __init__(
        self,
        rank: int,
        args: argparse.Namespace,
        comm_cfg: ConfigDict,
        hyper_params: ConfigDict,
        learner_state_dict: OrderedDict,
    ):
        ApeXWorker.__init__(self, rank, args, comm_cfg, hyper_params)
        self.loss_fn = build_loss(self.hyper_params.loss_type)

        self._init_networks(learner_state_dict)

    def _init_networks(self, state_dict: OrderedDict):
        self.dqn = torch.load(state_dict)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        # epsilon greedy policy

        # pylint: disable=comparison-with-callable
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        else:
            state = self._preprocess_state(state, self.device)
            selected_action = self.dqn(state).argmax()
            selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def compute_priorities(self, memory: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute initial priority values of experiences in local memory"""
        states = torch.FloatTensor(memory["states"]).to(self.device)
        actions = torch.LongTensor(memory["actions"]).to(self.device)
        rewards = torch.FloatTensor(memory["rewards"].reshape(-1, 1)).to(self.device)
        next_states = torch.FloatTensor(memory["next_states"]).to(self.device)
        dones = torch.FloatTensor(memory["dones"].reshape(-1, 1)).to(self.device)
        memory_tensors = (states, actions, rewards, next_states, dones)

        dq_loss_element_wise, _ = self.loss_fn(
            self.dqn, self.dqn, memory_tensors, self.hyper_params.gamma, self.head_cfg
        )
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params.per_eps

        return new_priorities

    def synchronize(self, new_params: List[np.ndarray]):
        self._synchronize(self.dqn, new_params)
