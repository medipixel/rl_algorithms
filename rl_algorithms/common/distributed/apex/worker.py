from abc import abstractmethod
import argparse
from collections import deque
from typing import Deque, Tuple

import numpy as np
import pyarrow as pa
import zmq
from zmq.sugar.context import Context

from rl_algorithms.common.distributed.abstract.worker import Worker
from rl_algorithms.utils.config import ConfigDict


# pylint: disable=abstract-method
class ApeXWorker(Worker):
    """Abstract class for ApeXWorker"""

    def __init__(
        self,
        rank: int,
        args: argparse.Namespace,
        comm_cfg: ConfigDict,
        hyper_params: ConfigDict,
        ctx: Context,
    ):
        Worker.__init__(self, rank, args, comm_cfg)
        self.hyper_params = hyper_params

        self._init_communication(ctx)

    def _init_communication(self, ctx: Context):
        """Initialize sockets connecting worker-learner, worker-buffer"""
        # for receiving params from learner
        self.sub_socket = ctx.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.CONFLATE, 1)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.learner_worker_port}")

        # for sending replay data to buffer
        self.push_socket = ctx.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.worker_buffer_port}")

    def collect_data(self) -> dict:
        """Fill and return local buffer"""
        local_memory = dict(states=[], actions=[], rewards=[], next_states=[], dones=[])
        local_memory_keys = local_memory.keys()
        if self.use_n_step:
            nstep_queue = deque(maxlen=self.hyper_params.n_step)

        while len(local_memory["states"]) < self.args.worker_buffer_max_size:
            state = self.env.reset()
            done = False
            score = 0

            while not done:
                if self.args.render and self.rank == 0:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                transition = (state, action, reward, next_state, done)

                if self.use_n_step:
                    nstep_queue.append(transition)
                    if self.hyper_params.n_step == len(nstep_queue):
                        nstep_exp = self.preprocess_nstep(
                            nstep_queue, self.hyper_params.gamma
                        )
                    for entry, keys in zip(nstep_exp, local_memory_keys):
                        local_memory[keys].append(entry)
                else:
                    for entry, keys in zip(transition, local_memory_keys):
                        local_memory[keys].append(entry)

                state = next_state
                score += reward

        for key in local_memory_keys:
            local_memory[key] = np.array(local_memory[key])

        return local_memory

    @staticmethod
    def preprocess_nstep(
        n_step_buffer: Deque[Tuple[np.ndarray, ...]], gamma: float
    ) -> Deque[Tuple[np.ndarray, ...]]:
        """Return n step transition"""
        # info of the last transition
        state, action = n_step_buffer[0][:2]
        reward, next_state, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s, d = transition[-3:]

            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_s, d) if d else (next_state, done)

        return state, action, reward, next_state, done

    @abstractmethod
    def compute_priorities(self, experience: Deque[Tuple[np.ndarray, ...]]):
        pass

    def send_data_to_buffer(self, experience):
        """Send replay data to global buffer"""
        replay_data_id = pa.serialize(experience).to_buffer()
        self.push_socket.send(replay_data_id)

    def recv_params_from_learner(self):
        """Get new params and sync. return True if success, False otherwise"""
        new_params_id = False
        try:
            new_params_id = self.sub_socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            return False

        if new_params_id:
            new_params = pa.deserialize(new_params_id)
            self.param_queue.append(new_params)
            self.synchronize(self.param_queue.pop())
        return True

    def run(self):
        """Run main worker loop"""
        while True:
            experience = self.collect_data()
            priority_values = self.compute_priorities(experience)
            worker_data = [experience, priority_values]
            self.send_data_to_buffer(worker_data)
            self.recv_params_from_learner()
