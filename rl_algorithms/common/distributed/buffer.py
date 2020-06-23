"""Wrapper for Ape-X global buffer

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

import argparse

import pyarrow as pa
import ray
import zmq

from rl_algorithms.common.abstract.buffer import BufferWrapper
from rl_algorithms.common.buffer.wrapper import PrioritizedBufferWrapper
from rl_algorithms.utils.config import ConfigDict


@ray.remote
class ApeXBufferWrapper(BufferWrapper):
    """Wrapper for Ape-X global buffer.

    Attributes:
        per_buffer (ReplayBuffer): prioritized replay buffer
        args (arpgarse.Namespace): args from run script
        hyper_params (ConfigDict): algorithm hyperparameters
        comm_config (ConfigDict): configs for communication

    """

    def __init__(
        self,
        per_buffer: PrioritizedBufferWrapper,
        args: argparse.Namespace,
        hyper_params: ConfigDict,
        comm_cfg: ConfigDict,
    ):
        BufferWrapper.__init__(self, per_buffer)
        self.args = args
        self.hyper_params = hyper_params
        self.comm_cfg = comm_cfg
        self.per_beta = hyper_params.per_beta
        self.num_sent = 0

    # pylint: disable=attribute-defined-outside-init
    def init_communication(self):
        """Initialize sockets for communication."""
        ctx = zmq.Context()
        self.req_socket = ctx.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.learner_buffer_port}")

        self.pull_socket = ctx.socket(zmq.PULL)
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.worker_buffer_port}")

    def recv_worker_data(self):
        """Receive replay data from worker and incorporate to buffer."""
        received = False
        try:
            new_replay_data_id = self.pull_socket.recv(zmq.DONTWAIT)
            received = True
        except zmq.Again:
            pass

        if received:
            new_replay_data = pa.deserialize(new_replay_data_id)
            experience, priorities = new_replay_data
            for idx in range(len(experience["states"])):
                transition = (
                    experience["states"][idx],
                    experience["actions"][idx],
                    experience["rewards"][idx],
                    experience["next_states"][idx],
                    experience["dones"][idx],
                )
                self.buffer.add(transition)
                self.buffer.update_priorities([len(self.buffer) - 1], priorities[idx])

    def send_batch_to_learner(self):
        """Send batch to learner and receive priorities."""
        # Send batch and request priorities (blocking recv)
        batch = self.buffer.sample(self.per_beta)
        batch_id = pa.serialize(batch).to_buffer()
        self.req_socket.send(batch_id)
        self.num_sent = self.num_sent + 1

        # Receive priorities
        new_priors_id = self.req_socket.recv()
        idxes, new_priorities = pa.deserialize(new_priors_id)
        self.buffer.update_priorities(idxes, new_priorities)

    def update_priority_beta(self):
        fraction = min(float(self.num_sent) / self.args.max_update_step, 1.0)
        self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

    def run(self):
        while self.num_sent < self.args.max_update_step:
            self.recv_worker_data()
            if len(self.buffer) >= self.hyper_params.update_starts_from:
                self.send_batch_to_learner()
                self.update_priority_beta()
