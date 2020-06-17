import argparse

import pyarrow as pa
import ray
import zmq

from rl_algorithms.common.abstract.buffer import BufferWrapper
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper
from rl_algorithms.utils.config import ConfigDict


@ray.remote
class ApeXBufferWrapper(BufferWrapper):
    def __init__(
        self,
        per_buffer: PrioritizedBufferWrapper,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        comm_cfg: ConfigDict,
    ):
        BufferWrapper.__init__(self, per_buffer)
        # self.per_buffer = per_buffer
        self.args = args
        self.hyper_params = hyper_params
        self.comm_cfg = comm_cfg
        self.per_beta = hyper_params.per_beta
        self.num_sent = 0

        self._init_communication()

    # pylint: disable=attribute-defined-outside-init
    def _init_communication(self):
        """Initialize sockets for communication"""
        ctx = zmq.Context()
        self.req_socket = ctx.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.learner_buffer_port}")

        self.pull_socket = ctx.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.worker_buffer_port}")

        # self.priorities_socket = ctx.socket(zmq.PULL)
        # self.priorities_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.priorities_port}")

        # self.send_batch_socket = ctx.socket(zmq.PUSH)
        # self.send_batch_socket.connect(
        #     f"tcp://127.0.0.1:{self.comm_cfg.send_batch_port}"
        # )

    def recv_worker_data(self):
        """Receive replay data from worker and incorporate to buffer."""
        new_replay_data_id = False
        try:
            new_replay_data_id = self.pull_socket.recv(zmq.DONTWAIT)
            return new_replay_data_id
        except zmq.Again:
            pass

    def send_batch_to_learner(self):
        """Send batch to learner and receive priorities"""
        # send batch and request priorities (blocking recv)
        batch = self.buffer.sample(self.per_beta)
        batch_id = pa.serialize(batch).to_buffer()
        self.req_socket.send(batch_id)
        self.num_sent = self.num_sent + 1

        # update priority beta
        self.update_priority_beta()

        new_priors_id = self.req_socket.recv()
        idxes, new_priorities = pa.deserialize(new_priors_id)
        self.buffer.update_priorities(idxes, new_priorities)

    def recv_priorities(self):
        # receive and update priorities
        new_priors_id = False
        try:
            new_priors_id = self.req_socket.recv(zmq.DONTWAIT)
            return new_priors_id
        except zmq.Again:
            pass

        # return True

    def update_priority_beta(self):
        fraction = min(float(self.num_sent) / self.args.max_update_step, 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

    def run(self):
        while True:
            new_replay_data_id = self.recv_worker_data()
            if new_replay_data_id is not None:
                new_replay_data = pa.deserialize(new_replay_data_id)

                if self.hyper_params.worker_computes_priorities:
                    experience, priorities = new_replay_data
                else:
                    experience = new_replay_data[0]

                for idx in range(len(experience["states"])):
                    transition = (
                        experience["states"][idx],
                        experience["actions"][idx],
                        experience["rewards"][idx],
                        experience["next_states"][idx],
                        experience["dones"][idx],
                    )
                    self.buffer.add(transition)
                    if self.hyper_params.worker_computes_priorities:
                        self.buffer.update_priorities(
                            [len(self.buffer) - 1], priorities[idx]
                        )
            if len(self.buffer) > self.hyper_params.update_starts_from:
                self.send_batch_to_learner()
                # new_priors_id = self.recv_priorities()
                # if new_priors_id is not None:
                #     idxes, new_priorities = pa.deserialize(new_priors_id)
                #     self.per_buffer.update_priorities(idxes, new_priorities)
