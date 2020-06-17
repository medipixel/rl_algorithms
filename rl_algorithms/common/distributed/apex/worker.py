from abc import abstractmethod
import argparse
from collections import deque
from typing import Deque, Dict

import numpy as np
import pyarrow as pa
import zmq

from rl_algorithms.common.distributed.abstract.worker import Worker
from rl_algorithms.utils.config import ConfigDict


# pylint: disable=abstract-method
class ApeXWorker(Worker):
    """Abstract class for ApeXWorker"""

    def __init__(
        self,
        rank: int,
        args: argparse.Namespace,
        env_info: ConfigDict,
        hyper_params: ConfigDict,
        comm_cfg: ConfigDict,
        device: str,
    ):
        Worker.__init__(self, rank, args, env_info, comm_cfg, device)
        self.hyper_params = hyper_params
        self.use_n_step = self.hyper_params.n_step > 1

        self._init_communication()
        self.epsilon = None

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

    def collect_data(self) -> dict:
        """Fill and return local buffer"""
        local_memory = dict(states=[], actions=[], rewards=[], next_states=[], dones=[])
        local_memory_keys = local_memory.keys()
        if self.use_n_step:
            nstep_queue = deque(maxlen=self.hyper_params.n_step)

        while len(local_memory["states"]) < self.hyper_params.local_buffer_max_size:
            state = self.env.reset()
            done = False
            score = 0
            num_steps = 0
            while not done:
                # if self.args.render:
                self.env.render()
                num_steps += 1
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                transition = (state, action, reward, next_state, int(done))
                if self.use_n_step:
                    nstep_queue.append(transition)
                    if self.hyper_params.n_step == len(nstep_queue):
                        nstep_exp = self.preprocess_data(nstep_queue)
                        for entry, keys in zip(nstep_exp, local_memory_keys):
                            local_memory[keys].append(entry)
                else:
                    for entry, keys in zip(transition, local_memory_keys):
                        local_memory[keys].append(entry)

                new_params_id = self.recv_params_from_learner()
                if new_params_id is not None:
                    new_params = pa.deserialize(new_params_id)
                    self.synchronize(new_params)

                state = next_state
                score += reward

            print(score, num_steps, self.epsilon)

        for key in local_memory_keys:
            local_memory[key] = np.array(local_memory[key])

        return local_memory

    @abstractmethod
    def compute_priorities(self, experience: Dict[str, np.ndarray]):
        pass

    def send_data_to_buffer(self, replay_data):
        """Send replay data to global buffer"""
        replay_data_id = pa.serialize(replay_data).to_buffer()
        self.push_socket.send(replay_data_id)

    def recv_params_from_learner(self):
        """Get new params and sync. return True if success, False otherwise"""
        new_params_id = False
        try:
            new_params_id = self.sub_socket.recv(zmq.DONTWAIT)
            return new_params_id
        except zmq.Again:
            return False

    def run(self):
        """Run main worker loop"""
        while True:
            experience = self.collect_data()
            worker_data = [experience]
            if self.hyper_params.worker_computes_priorities:
                priority_values = self.compute_priorities(experience)
                worker_data.append(priority_values)
            self.send_data_to_buffer(worker_data)
            # new_params_id = self.recv_params_from_learner()
            # if new_params_id is not None:
            #     new_params = pa.deserialize(new_params_id)
            #     # print(new_params)
            #     self.synchronize(new_params)

    def preprocess_data(self, nstepqueue: Deque) -> tuple:
        discounted_reward = 0
        _, _, _, last_state, done = nstepqueue[-1]
        for transition in list(reversed(nstepqueue)):
            state, action, reward, _, _ = transition
            discounted_reward = reward + self.hyper_params.gamma * discounted_reward
        nstep_data = (state, action, discounted_reward, last_state, done)

        # q_value = self.brain.forward(
        #     torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # )[0][action]

        # bootstrap_q = torch.max(
        #     self.brain.forward(
        #         torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
        #     ),
        #     1,
        # )

        # target_q_value = (
        #     discounted_reward + self.gamma ** self.num_step * bootstrap_q[0]
        # )

        # priority_value = torch.abs(target_q_value - q_value).detach().view(-1)
        # priority_value = torch.clamp(priority_value, min=1e-8)
        # priority_value = priority_value.cpu().numpy().tolist()

        return nstep_data  # , priority_value
