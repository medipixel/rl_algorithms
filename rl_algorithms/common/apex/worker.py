"""Wrapper class for ApeX based distributed workers.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""
from collections import deque
from typing import Dict

import numpy as np
import pyarrow as pa
import ray
import zmq

from rl_algorithms.common.abstract.distributed_worker import (
    DistributedWorker,
    DistributedWorkerWrapper,
)
from rl_algorithms.utils.config import ConfigDict


@ray.remote(num_cpus=1)
class ApeXWorkerWrapper(DistributedWorkerWrapper):
    """Wrapper class for ApeX based distributed workers.

    Attributes:
        hyper_params (ConfigDict): worker hyper_params
        update_step (int): tracker for learner update step
        use_n_step (int): indication for using n-step transitions
        sub_socket (zmq.Context): subscriber socket for receiving params from learner
        push_socket (zmq.Context): push socket for sending experience to global buffer

    """

    def __init__(self, worker: DistributedWorker, comm_cfg: ConfigDict):
        DistributedWorkerWrapper.__init__(self, worker, comm_cfg)
        self.update_step = 0
        self.hyper_params = self.worker.hyper_params
        self.use_n_step = self.hyper_params.n_step > 1
        self.scores = dict()

        self.worker._init_env()

    # pylint: disable=attribute-defined-outside-init
    def init_communication(self):
        """Initialize sockets connecting worker-learner, worker-buffer."""
        # for receiving params from learner
        ctx = zmq.Context()
        self.sub_socket = ctx.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub_socket.setsockopt(zmq.RCVHWM, 2)
        self.sub_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.learner_worker_port}")

        # for sending replay data to buffer
        self.push_socket = ctx.socket(zmq.PUSH)
        self.push_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.worker_buffer_port}")

    def send_data_to_buffer(self, replay_data):
        """Send replay data to global buffer."""
        replay_data_id = pa.serialize(replay_data).to_buffer()
        self.push_socket.send(replay_data_id)

    def recv_params_from_learner(self):
        """Get new params and sync. return True if success, False otherwise."""
        received = False
        try:
            new_params_id = self.sub_socket.recv(zmq.DONTWAIT)
            received = True
        except zmq.Again:
            # Although learner doesn't send params, don't wait
            pass

        if received:
            new_param_info = pa.deserialize(new_params_id)
            update_step, new_state_dict = new_param_info
            self.update_step = update_step
            self.worker.synchronize(new_state_dict)

            # Add new entry for scores dict
            self.scores[self.update_step] = []

    def compute_priorities(self, experience: Dict[str, np.ndarray]):
        """Compute priority values (TD error) of collected experience."""
        return self.worker.compute_priorities(experience)

    def collect_data(self) -> dict:
        """Fill and return local buffer."""
        local_memory = dict(states=[], actions=[], rewards=[], next_states=[], dones=[])
        local_memory_keys = local_memory.keys()
        if self.use_n_step:
            nstep_queue = deque(maxlen=self.hyper_params.n_step)

        while len(local_memory["states"]) < self.hyper_params.local_buffer_max_size:
            state = self.worker.env.reset()
            done = False
            score = 0
            num_steps = 0
            while not done:
                if self.hyper_params.is_worker_render:
                    self.worker.env.render()
                num_steps += 1
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                transition = (state, action, reward, next_state, int(done))
                if self.use_n_step:
                    nstep_queue.append(transition)
                    if self.hyper_params.n_step == len(nstep_queue):
                        nstep_exp = self.preprocess_nstep(nstep_queue)
                        for entry, keys in zip(nstep_exp, local_memory_keys):
                            local_memory[keys].append(entry)
                else:
                    for entry, keys in zip(transition, local_memory_keys):
                        local_memory[keys].append(entry)

                state = next_state
                score += reward

                self.recv_params_from_learner()

            self.scores[self.update_step].append(score)

            if self.hyper_params.is_worker_log:
                print(
                    f"[TRAIN] [Worker {self.worker.rank}] "
                    + f"Update step: {self.update_step}, Score: {score}, "
                    + f"Epsilon: {self.worker.epsilon:.5f}"
                )

        for key in local_memory_keys:
            local_memory[key] = np.array(local_memory[key])

        return local_memory

    def run(self) -> Dict[int, float]:
        """Run main worker loop."""
        self.scores[self.update_step] = []
        while self.update_step < self.hyper_params.max_update_step:
            experience = self.collect_data()
            priority_values = self.compute_priorities(experience)
            worker_data = [experience, priority_values]
            self.send_data_to_buffer(worker_data)

        mean_scores_per_ep_step = self.compute_mean_scores(self.scores)
        return mean_scores_per_ep_step

    @staticmethod
    def compute_mean_scores(scores: Dict[int, list]):
        for step in list(scores):
            if scores[step]:
                scores[step] = np.mean(scores[step])
            else:
                # Delete empty score list
                # made when network is updated before termination of episode
                scores.pop(step)
        return scores
