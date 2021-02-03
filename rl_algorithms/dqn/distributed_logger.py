"""DQN Logger for distributed training.

- Author: Chris Yoon
- Contact: chris.yoon@medipixel.io
"""

import numpy as np
import torch
import wandb

from rl_algorithms.common.abstract.distributed_logger import DistributedLogger
from rl_algorithms.registry import LOGGERS


@LOGGERS.register_module
class DQNLogger(DistributedLogger):
    """DQN Logger for distributed training."""

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        # Logger only runs on cpu
        DistributedLogger.load_params(self, path)

        params = torch.load(path, map_location="cpu")
        self.brain.load_state_dict(params["dqn_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def select_action(self, state: np.ndarray):
        """Select action to be executed at given state."""
        with torch.no_grad():
            state = self._preprocess_state(state, self.device)
            selected_action = self.brain(state).argmax()
        selected_action = selected_action.cpu().numpy()

        return selected_action

    def write_log(self, log_value: dict):
        """Write log about loss and score."""
        print(
            "[INFO] update_step %d, average score: %f, "
            "loss: %f, avg q-value: %f"
            % (
                log_value["update_step"],
                log_value["avg_score"],
                log_value["step_info"][0],
                log_value["step_info"][1],
            )
        )

        if self.is_log:
            wandb.log(
                {
                    "test score": log_value["avg_score"],
                    "dqn loss": log_value["step_info"][0],
                    "avg q values": log_value["step_info"][1],
                },
                step=log_value["update_step"],
            )
