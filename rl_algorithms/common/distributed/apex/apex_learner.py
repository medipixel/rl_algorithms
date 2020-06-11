import time

import numpy as np
import pyarrow as pa
import zmq

from rl_algorithms.common.abstract.learner import Learner
from rl_algorithms.common.distributed.abstract.distributed_learner import (
    DistributedLearner,
)
from rl_algorithms.common.helper_functions import params2numpy
from rl_algorithms.utils.config import ConfigDict

# TODO: Add communication with logger process


class ApeXLearner(DistributedLearner):
    """Learner Wrapper to enable Ape-X distributed training

    Attributes:
        learner (Learner): learner
        comm_config (ConfigDict): configs for communication
        update_step (int): counts update steps
        pub_socket (zmq.Socket): publisher socket for broadcasting params
        rep_socket (zmq.Socket): reply socket for receiving replay data & sending new priorities

    """

    def __init__(self, learner: Learner, comm_cfg: ConfigDict):
        DistributedLearner.__init__(self, learner, comm_cfg)
        self.update_step = 0

        self._init_network()
        self._init_communication()

    # pylint: disable=attribute-defined-outside-init
    def _init_communication(self):
        """Initialize sockets for communication"""
        ctx = zmq.Context()
        self.pub_socket = ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg['learner_worker_port']}")

        ctx = zmq.Context()
        self.rep_socket = ctx.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg['learner_buffer_port']}")

    def publish_params(self):
        """Broadcast updated params to all workers"""
        new_params = self.get_policy()
        new_params = params2numpy(new_params)
        new_params_id = pa.serialize(new_params).to_buffer()
        self.pub_socket.send(new_params_id)

    def recv_replay_data(self):
        """Receive replay data from buffer"""
        try:
            replay_data_id = self.rep_socket.recv()
        except zmq.Again:
            return False

        replay_data = pa.deserialize(replay_data_id)
        return replay_data

    def send_new_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Send new priority values and corresponding indices to buffer"""
        new_priors = [indices, priorities]
        new_priors_id = pa.deserialize(new_priors).to_buffer()
        self.rep_socket.send(new_priors_id)

    def run(self):
        """Run main training loop"""
        # wait for a bit
        time.sleep(3)

        while True:
            replay_data = self.recv_replay_data()
            if replay_data:
                info = self.update_model(replay_data)
                indices, new_priorities = info[-3:-1]
                self.send_new_priorities(indices, new_priorities)
                self.update_step = self.update_step + 1

            if self.update_step % self.worker_update_interval == 0:
                self.publish_params()

            if self.update_step == self.max_update_step:
                # finish training
                break
