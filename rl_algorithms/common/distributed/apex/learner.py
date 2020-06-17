from typing import List

import numpy as np
import pyarrow as pa
import ray
import torch
import zmq

from rl_algorithms.common.abstract.learner import Learner
from rl_algorithms.common.distributed.abstract.distributed_learner import (
    DistributedLearnerWrapper,
)
from rl_algorithms.common.helper_functions import numpy2floattensor, state_dict2numpy
from rl_algorithms.utils.config import ConfigDict

# TODO: Add communication with logger process


@ray.remote(num_gpus=1)
class ApeXLearnerWrapper(DistributedLearnerWrapper):
    """Learner Wrapper to enable Ape-X distributed training

    Attributes:
        learner (Learner): learner
        comm_config (ConfigDict): configs for communication
        update_step (int): counts update steps
        pub_socket (zmq.Socket): publisher socket for broadcasting params
        rep_socket (zmq.Socket): reply socket for receiving replay data & sending new priorities

    """

    def __init__(self, learner: Learner, comm_cfg: ConfigDict):
        DistributedLearnerWrapper.__init__(self, learner, comm_cfg)
        self.update_step = 0
        self.max_update_step = self.learner.args.max_update_step
        self.worker_update_interval = self.learner.hyper_params.worker_update_interval
        # disable; learner uses preprocessed n_step experience
        self.learner.use_n_step = False

        self._init_network()
        self._init_communication()

    # pylint: disable=attribute-defined-outside-init
    def _init_communication(self):
        """Initialize sockets for communication"""
        ctx = zmq.Context()
        self.pub_socket = ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.learner_worker_port}")

        self.rep_socket = ctx.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.learner_buffer_port}")

        # self.push_socket = ctx.socket(zmq.PUSH)
        # self.push_socket.connect(f"tcp://127.0.0.1:{self.comm_cfg.learner_logger_port}")

        # self.priorities_socket = ctx.socket(zmq.PUSH)
        # self.priorities_socket.connect(
        #     f"tcp://127.0.0.1:{self.comm_cfg.priorities_port}"
        # )

        # self.batch_socket = ctx.socket(zmq.PULL)
        # self.batch_socket.bind(f"tcp://127.0.0.1:{self.comm_cfg.send_batch_port}")

    def publish_params(self, np_state_dict: List[np.ndarray]):
        """Broadcast updated params to all workers"""
        new_params_id = pa.serialize(np_state_dict).to_buffer()
        self.pub_socket.send(new_params_id)

    def recv_replay_data(self):
        replay_data_id = self.rep_socket.recv()
        replay_data = pa.deserialize(replay_data_id)
        # self.replay_data_queue.append(replay_data)
        return replay_data

    def send_new_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Send new priority values and corresponding indices to buffer"""
        new_priors = [indices, priorities]
        new_priors_id = pa.serialize(new_priors).to_buffer()
        self.rep_socket.send(new_priors_id)

    def send_info_to_logger(self, new_params: List[np.ndarray], info: list, telapsed):
        log_value = dict(update_step=self.update_step, loss=info, time_elapsed=telapsed)
        log_info = dict(log_value=log_value, state_dict=new_params)
        log_info_id = pa.serialize(log_info).to_buffer()
        self.push_socket.send(log_info_id)

    def run(self):
        """Run main training loop"""
        # wait for a bit
        self.telapsed = 0
        while True:
            replay_data = self.recv_replay_data()
            if replay_data is not None:
                replay_data = numpy2floattensor(replay_data)
                info = self.update_model(replay_data)

                indices, new_priorities = info[-2:]
                self.update_step = self.update_step + 1

                self.send_new_priorities(indices, new_priorities)

                if self.update_step % self.worker_update_interval == 0:
                    print("publishing...")
                    state_dict = self.get_state_dict()
                    np_state_dict = state_dict2numpy(state_dict)
                    self.publish_params(np_state_dict)
                    # self.send_info_to_logger(np_state_dict, info, str(self.telapsed))

            if self.update_step == self.max_update_step:
                # finish training
                break

    def select_action(self, state: np.ndarray, brain):
        state = self._preprocess_state(state, self.learner.device)
        selected_action = brain(state).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    @staticmethod
    def _preprocess_state(state: np.ndarray, device: torch.device) -> torch.Tensor:
        state = torch.FloatTensor(state).to(device)
        return state
