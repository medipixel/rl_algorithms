from abc import ABC, abstractmethod
import argparse
from collections import deque
import os
import shutil
from typing import List

import gym
import numpy as np
import pyarrow as pa
import torch
import wandb
import zmq

from rl_algorithms.common.env.atari_wrappers import atari_env_generator
import rl_algorithms.common.env.utils as env_utils
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class Logger(ABC):
    def __init__(
        self,
        args: argparse.Namespace,
        env_info: ConfigDict,
        log_cfg: ConfigDict,
        comm_cfg: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
    ):
        self.args = args
        self.env_info = env_info
        self.log_cfg = log_cfg
        self.comm_cfg = comm_cfg
        self.device = torch.device("cpu")
        self.brain = Brain(backbone, head).to(self.device)

        self.log_info_queue = deque(maxlen=1000)

        self._init_communication()
        self._init_env()

    def _init_env(self):
        if self.env_info.is_atari:
            self.env = atari_env_generator(
                self.env_info.name, self.args.max_episode_steps
            )
        else:
            self.env = gym.make(self.env_info.name)
            env_utils.set_env(self.env, self.args)

    def _init_communication(self):
        ctx = zmq.Context()
        self.pull_socket = ctx.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.learner_logger_port}")

    @abstractmethod
    def select_action(self, state: np.ndarray):
        pass

    @abstractmethod
    def write_log(self, log_value: dict):
        pass

    # pylint: disable=no-self-use
    @staticmethod
    def _preprocess_state(state: np.ndarray, device: torch.device) -> torch.Tensor:
        state = torch.FloatTensor(state).to(device)
        return state

    @abstractmethod
    def synchronize(self, new_params: list):
        pass

    # pylint: disable=no-self-use
    def _synchronize(self, network, new_params: List[np.ndarray]):
        """Copy parameters from numpy arrays"""
        for param, new_param in zip(network.parameters(), new_params):
            new_param = torch.FloatTensor(new_param).to(self.device)
            param.data.copy_(new_param)

    def set_wandb(self):
        """Set configuration for wandb logging."""
        wandb.init(
            project=self.log_cfg.env_name,
            name=f"{self.log_cfg.agent}/{self.log_cfg.curr_time}",
        )
        wandb.config.update(vars(self.args))
        shutil.copy(self.args.cfg_path, os.path.join(wandb.run.dir, "config.py"))

    def recv_log_info(self):
        log_info_id = False
        try:
            log_info_id = self.pull_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if log_info_id:
            self.log_info_queue.append(log_info_id)

        return True

    def run(self):
        # logger
        if self.args.log:
            self.set_wandb()

        while True:
            self.recv_log_info()
            if not self.log_info_queue.empty():
                log_info_id = self.log_info_queue.pop()
                log_info = pa.deserialize(log_info_id)
                log_value = log_info["log_value"]
                state_dict = log_info["state_dict"]

                self.synchronize(state_dict)
                avg_score = self.test(log_value["update_step"])
                log_value["avg_score"] = avg_score

                self.write_log(log_value)

    def test(self, update_step: int):
        """Test the agent."""
        avg_score = self._test(update_step)

        # termination
        self.env.close()

        return avg_score

    def _test(self, update_step: int, interim_test: bool = True) -> float:
        """Common test routine."""
        if interim_test:
            test_num = self.args.interim_test_num
        else:
            test_num = self.args.episode_num

        scores = []
        for i_episode in range(test_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                # if self.args.render:
                #     self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                state = next_state
                score += reward
                step += 1

            scores.append(score)

            print(
                "[INFO] update step: %d\ttest %d\tstep: %d\ttotal score: %d"
                % (update_step, i_episode, step, score)
            )

        return np.mean(scores)
