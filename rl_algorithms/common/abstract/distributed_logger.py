"""Base class for loggers use in distributed training.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

from abc import ABC, abstractmethod
from collections import deque
import os
import shutil
from typing import Dict, List

import gym
import numpy as np
import plotly.graph_objects as go
import pyarrow as pa
import torch
import wandb
import zmq

from rl_algorithms.common.env.atari_wrappers import atari_env_generator
import rl_algorithms.common.env.utils as env_utils
from rl_algorithms.common.helper_functions import numpy2floattensor, smoothen_graph
from rl_algorithms.common.networks.brain import Brain
from rl_algorithms.utils.config import ConfigDict


class DistributedLogger(ABC):
    """Base class for loggers use in distributed training.

    Attributes:
        log_cfg (ConfigDict): configuration for saving log and checkpoint
        comm_config (ConfigDict): configs for communication
        backbone (ConfigDict): backbone configs for building network
        head (ConfigDict): head configs for building network
        brain (Brain): logger brain for evaluation
        update_step (int): tracker for learner update step
        device (torch.device): device, cpu by default
        log_info_queue (deque): queue for storing log info received from learner
        env (gym.Env): gym environment for running test

    """

    def __init__(
        self,
        log_cfg: ConfigDict,
        comm_cfg: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
        env_name: str,
        is_atari: bool,
        state_size: int,
        output_size: int,
        max_update_step: int,
        episode_num: int,
        max_episode_steps: int,
        interim_test_num: int,
        is_log: bool,
        is_render: bool,
    ):
        self.log_cfg = log_cfg
        self.comm_cfg = comm_cfg
        self.device = torch.device("cpu")  # Logger only runs on cpu
        head.configs.state_size = state_size
        head.configs.output_size = output_size
        self.brain = Brain(backbone, head).to(self.device)

        self.env_name = env_name
        self.is_atari = is_atari
        self.max_update_step = max_update_step
        self.episode_num = episode_num
        self.max_episode_steps = max_episode_steps
        self.interim_test_num = interim_test_num
        self.is_log = is_log
        self.is_render = is_render

        self.update_step = 0
        self.log_info_queue = deque(maxlen=100)

        self._init_env()

    # pylint: disable=attribute-defined-outside-init
    def _init_env(self):
        """Initialize gym environment."""
        if self.is_atari:
            self.env = atari_env_generator(self.env_name, self.max_episode_steps)
        else:
            self.env = gym.make(self.env_name)
            self.env, self.max_episode_steps = env_utils.set_env(
                self.env, self.max_episode_steps
            )

    @abstractmethod
    def load_params(self, path: str):
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )

    # pylint: disable=attribute-defined-outside-init
    def init_communication(self):
        """Initialize inter-process communication sockets."""
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
        state = numpy2floattensor(state, device)
        return state

    def set_wandb(self):
        """Set configuration for wandb logging."""
        wandb.init(
            project=self.env_name,
            name=f"{self.log_cfg.agent}/{self.log_cfg.curr_time}",
        )
        additional_log = dict(
            episode_num=self.episode_num,
            max_episode_steps=self.max_episode_steps,
        )
        wandb.config.update(additional_log)
        shutil.copy(self.log_cfg.cfg_path, os.path.join(wandb.run.dir, "config.yaml"))

    def recv_log_info(self):
        """Receive info from learner."""
        received = False
        try:
            log_info_id = self.pull_socket.recv(zmq.DONTWAIT)
            received = True
        except zmq.Again:
            pass

        if received:
            self.log_info_queue.append(log_info_id)

    def run(self):
        """Run main logging loop; continuously receive data and log."""
        if self.is_log:
            self.set_wandb()

        while self.update_step < self.max_update_step:
            self.recv_log_info()
            if self.log_info_queue:  # if non-empty
                log_info_id = self.log_info_queue.pop()
                log_info = pa.deserialize(log_info_id)
                state_dict = log_info["state_dict"]
                log_value = log_info["log_value"]
                self.update_step = log_value["update_step"]

                self.synchronize(state_dict)
                avg_score = self.test(self.update_step)
                log_value["avg_score"] = avg_score
                self.write_log(log_value)

    def write_worker_log(self, worker_logs: List[dict], worker_update_interval: int):
        """Log the mean scores of each episode per update step to wandb."""
        # NOTE: Worker plots are passed onto wandb.log as matplotlib.pyplot
        #       since wandb doesn't support logging multiple lines to single plot
        self.set_wandb()
        # Plot individual workers
        fig = go.Figure()
        worker_id = 0
        for worker_log in worker_logs:
            fig.add_trace(
                go.Scatter(
                    x=list(worker_log.keys()),
                    y=smoothen_graph(list(worker_log.values())),
                    mode="lines",
                    name=f"Worker {worker_id}",
                    line=dict(width=2),
                )
            )
            worker_id = worker_id + 1

        # Plot mean scores
        logged_update_steps = list(
            range(0, self.max_update_step + 1, worker_update_interval)
        )

        mean_scores = []
        try:
            for step in logged_update_steps:
                scores_for_step = []
                for worker_log in worker_logs:
                    if step in list(worker_log):
                        scores_for_step.append(worker_log[step])
                mean_scores.append(np.mean(scores_for_step))
        except Exception as e:
            print(f"[Error] {e}")

        fig.add_trace(
            go.Scatter(
                x=logged_update_steps,
                y=mean_scores,
                mode="lines+markers",
                name="Mean scores",
                line=dict(width=5),
            )
        )

        # Write to wandb
        wandb.log({"Worker scores": fig})

    def test(self, update_step: int, interim_test: bool = True):
        """Test the agent."""
        avg_score = self._test(update_step, interim_test)

        # termination
        self.env.close()
        return avg_score

    def _test(self, update_step: int, interim_test: bool) -> float:
        """Common test routine."""
        if interim_test:
            test_num = self.interim_test_num
        else:
            test_num = self.episode_num

        self.brain.eval()
        scores = []
        for i_episode in range(test_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.is_render:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)

                state = next_state
                score += reward
                step += 1

            scores.append(score)

            if interim_test:
                print(
                    "[INFO] update step: %d\ttest %d\tstep: %d\ttotal score: %d"
                    % (update_step, i_episode, step, score)
                )
            else:
                print(
                    "[INFO] test %d\tstep: %d\ttotal score: %d"
                    % (i_episode, step, score)
                )

        return np.mean(scores)

    def synchronize(self, state_dict: Dict[str, np.ndarray]):
        """Copy parameters from numpy arrays."""
        param_name_list = list(state_dict.keys())
        for logger_named_param in self.brain.named_parameters():
            logger_param_name = logger_named_param[0]
            if logger_param_name in param_name_list:
                new_param = numpy2floattensor(
                    state_dict[logger_param_name], self.device
                )
                logger_named_param[1].data.copy_(new_param)
