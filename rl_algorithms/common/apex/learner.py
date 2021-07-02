"""Learner Wrapper to enable Ape-X distributed training.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

from typing import Dict, List

import numpy as np
import pyarrow as pa
import ray
import zmq

from rl_algorithms.common.abstract.learner import DistributedLearnerWrapper, Learner
from rl_algorithms.common.helper_functions import numpy2floattensor, state_dict2numpy
from rl_algorithms.utils.config import ConfigDict


@ray.remote(num_gpus=1)
class ApeXLearnerWrapper(DistributedLearnerWrapper):
    """Learner Wrapper to enable Ape-X distributed training.

    Attributes:
        learner (Learner): learner
        comm_config (ConfigDict): configs for communication
        update_step (int): counts update steps
        pub_socket (zmq.Socket): publisher socket for broadcasting params
        rep_socket (zmq.Socket): reply socket for receiving replay data & sending new priorities
        update_step (int): number of update steps
        max_update_step (int): maximum update steps per run
        worker_update_interval (int): num update steps between worker synchronization
        logger_interval (int): num update steps between logging

    """

    def __init__(self, learner: Learner, comm_cfg: ConfigDict):
        """Initialize."""
        DistributedLearnerWrapper.__init__(self, learner, comm_cfg)
        self.update_step = 0
        self.max_update_step = self.learner.hyper_params.max_update_step
        self.worker_update_interval = self.learner.hyper_params.worker_update_interval
        self.logger_interval = self.learner.hyper_params.logger_interval

        # NOTE: disable because learner uses preprocessed n_step experience
        self.learner.use_n_step = False

    # pylint: disable=attribute-defined-outside-init
    def init_communication(self):
        """Initialize sockets for communication."""
        ctx = zmq.Context()
        # Socket to send updated network parameters to worker
        self.pub_socket = ctx.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 2)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.learner_worker_port}")

        # Socket to receive replay data and send new priorities to buffer
        self.rep_socket = ctx.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.learner_buffer_port}")

        # Socket to send logging data to logger
        self.push_socket = ctx.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.learner_logger_port}")

    def recv_replay_data(self):
        """Receive replay data from gloal buffer."""
        replay_data_id = self.rep_socket.recv()
        replay_data = pa.deserialize(replay_data_id)
        return replay_data

    def send_new_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Send new priority values and corresponding indices to buffer."""
        new_priors = [indices, priorities]
        new_priors_id = pa.serialize(new_priors).to_buffer()
        self.rep_socket.send(new_priors_id)

    def publish_params(self, update_step: int, np_state_dict: Dict[str, np.ndarray]):
        """Broadcast updated params to all workers."""
        param_info = [update_step, np_state_dict]
        new_params_id = pa.serialize(param_info).to_buffer()
        self.pub_socket.send(new_params_id)

    def send_info_to_logger(
        self,
        np_state_dict: List[np.ndarray],
        step_info: list,
    ):
        """Send new params and log info to logger."""
        log_value = dict(update_step=self.update_step, step_info=step_info)
        log_info = dict(log_value=log_value, state_dict=np_state_dict)
        log_info_id = pa.serialize(log_info).to_buffer()
        self.push_socket.send(log_info_id)

    def run(self):
        """Run main training loop."""
        self.telapsed = 0
        while self.update_step < self.max_update_step:
            replay_data = self.recv_replay_data()
            if replay_data is not None:
                replay_data = (
                    numpy2floattensor(replay_data[:6], self.learner.device)
                    + replay_data[6:]
                )
                info = self.update_model(replay_data)
                indices, new_priorities = info[-2:]
                step_info = info[:-2]
                self.update_step = self.update_step + 1

                self.send_new_priorities(indices, new_priorities)

                if self.update_step % self.worker_update_interval == 0:
                    state_dict = self.get_state_dict()
                    np_state_dict = state_dict2numpy(state_dict)
                    self.publish_params(self.update_step, np_state_dict)

                if self.update_step % self.logger_interval == 0:
                    state_dict = self.get_state_dict()
                    np_state_dict = state_dict2numpy(state_dict)
                    self.send_info_to_logger(np_state_dict, step_info)
                    self.learner.save_params(self.update_step)
