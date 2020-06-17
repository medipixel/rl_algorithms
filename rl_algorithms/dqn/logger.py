import argparse
from typing import List

import numpy as np
import wandb

from rl_algorithms.common.distributed.abstract.logger import Logger
from rl_algorithms.registry import LOGGERS
from rl_algorithms.utils.config import ConfigDict


@LOGGERS.register_module
class DQNLogger(Logger):
    def __init__(
        self,
        args: argparse.Namespace,
        env_info: ConfigDict,
        log_cfg: ConfigDict,
        comm_cfg: ConfigDict,
        backbone: ConfigDict,
        head: ConfigDict,
    ):
        Logger.__init__(self, args, env_info, log_cfg, comm_cfg, backbone, head)

    def select_action(self, state: np.ndarray):
        state = self._preprocess_state(state, self.device)
        selected_action = self.brain(state).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def write_log(self, log_value: dict):
        """Write log about loss and score"""
        print(
            "[INFO] update_step %d, average score: %f \n"
            "loss: %f, avg q-value: %f time elapsed: %s\n"
            % (
                log_value["update_step"],
                log_value["avg_score"],
                log_value["loss"][0],
                log_value["loss"][1],
                log_value["time_elapsed"],
            )
        )

        if self.args.log:
            wandb.log(
                {log_value["avg_score"], log_value["loss"][0], log_value["loss"][1]}
            )

    def synchronize(self, new_params: List[np.ndarray]):
        self._synchronize(self.brain, new_params)
