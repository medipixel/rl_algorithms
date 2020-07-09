# -*- coding: utf-8 -*-
"""Distillation buffer."""

import os
import pickle
from typing import Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DistillationBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
        obs_buf (np.ndarray): observations
        q_value_buf (np.ndarray): q_values for distillation
        buffer_size (int): size of buffers
        batch_size (int): batch size for training
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition

    """

    def __init__(
        self, batch_size: int, buffer_path, curr_time,
    ):
        """Initialize a DistillationBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training

        """
        self.batch_size = batch_size
        self.buffer_path = buffer_path
        self.idx = 0
        self.buffer_size = len(next(os.walk(self.buffer_path))[2])
        self.curr_time = curr_time
        self.dataloader = None

    def add(self, transition: Tuple[np.ndarray, np.ndarray]) -> Tuple[Any, ...]:
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        file_path = os.path.join(
            self.buffer_path,
            self.curr_time + "_transition_step_" + str(self.idx) + ".pkl",
        )
        with open(file_path, "wb") as f:
            pickle.dump(transition, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.idx += 1

    def reset_dataloader(self):
        dataset = DistillationDataset(self.buffer_path, self.buffer_size)
        self.dataloader = iter(
            DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        )

    def sample_for_diltillation(self):
        """Sample a batch of state and Q-value for policy distillation."""
        assert self.buffer_size >= self.batch_size

        return next(self.dataloader)


class DistillationDataset(Dataset):
    def __init__(self, buffer_path, buffer_size):
        self.buffer_path = buffer_path
        self.buffer_length = buffer_size

    def __len__(self):
        "Denotes the total number of samples"
        return self.buffer_length

    def __getitem__(self, index):
        "Generates one sample of data"
        file_name = os.listdir(self.buffer_path)[index]
        file_path = os.path.join(self.buffer_path, file_name)
        with open(file_path, "rb") as f:
            transition = pickle.load(f)
        return transition
