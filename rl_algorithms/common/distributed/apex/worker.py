import argparse
from typing import Dict

import numpy as np
import pyarrow as pa
import ray
import zmq

from rl_algorithms.common.distributed.abstract.worker import (
    DistributedWorkerWrapper,
    Worker,
)
from rl_algorithms.utils.config import ConfigDict


@ray.remote(num_cpus=1)
class ApeXWorkerWrapper(DistributedWorkerWrapper):
    """Wrapper class for ApeX based distributed workers"""

    def __init__(self, worker: Worker, args: argparse.Namespace, comm_cfg: ConfigDict):
        DistributedWorkerWrapper.__init__(self, worker, args, comm_cfg)
        self.update_step = 0

        self.worker._init_env()
        self._init_communication()

    # pylint: disable=attribute-defined-outside-init
    def _init_communication(self):
        """Initialize sockets connecting worker-learner, worker-buffer"""
        # for receiving params from learner
        ctx = zmq.Context()
        self.sub_socket = ctx.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.learner_worker_port}")

        # for sending replay data to buffer
        self.push_socket = ctx.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.worker_buffer_port}")

    def send_data_to_buffer(self, replay_data):
        """Send replay data to global buffer"""
        replay_data_id = pa.serialize(replay_data).to_buffer()
        self.push_socket.send(replay_data_id)

    def recv_params_from_learner(self):
        """Get new params and sync. return True if success, False otherwise"""
        received = False
        try:
            new_params_id = self.sub_socket.recv(zmq.DONTWAIT)
            received = True
        except zmq.Again:
            pass

        if received:
            new_param_info = pa.deserialize(new_params_id)
            update_step, new_params = new_param_info
            self.update_step = update_step
            self.worker.synchronize(new_params)

    def compute_priorities(self, experience: Dict[str, np.ndarray]):
        return self.worker.compute_priorities(experience)

    def run(self):
        """Run main worker loop"""
        while self.update_step < self.args.max_update_step:
            experience = self.collect_data()
            priority_values = self.compute_priorities(experience)
            worker_data = [experience, priority_values]
            self.send_data_to_buffer(worker_data)
            self.recv_params_from_learner()
