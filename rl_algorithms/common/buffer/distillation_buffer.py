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
    """Class for managing reading and writing of distillation data.
       Distillation data is stored in the buffer_path location on the actual hard disk.
       The data collected by the teacher is stored as individual pickle files.
       It is also read as batch through the pytorch DataLoader class.

    Attributes:
        batch_size (int): size of batch size from distillation buffer for training
        buffer_path (str): distillation buffer path
        curr_time (str): program's start time to distinguish between teacher agents
        idx (int): index of data
        buffer_size (int): distillation buffer size
        dataloader (DataLoader): pytorch library for random batch data sampling

    """

    def __init__(
        self, batch_size: int, buffer_path: str, curr_time: str,
    ):
        """Initialize a DistillationBuffer object.

        Args:
            batch_size (int): size of a batched sampled from distillation buffer for training
            buffer_path (str): distillation buffer path
            curr_time (str): program's start time to distinguish between teacher agents

        """
        self.batch_size = batch_size
        self.buffer_path = buffer_path
        self.idx = 0
        self.buffer_size = len(next(os.walk(self.buffer_path))[2])
        self.curr_time = curr_time
        self.dataloader = None

    def add(self, data: Tuple[np.ndarray, np.ndarray]) -> Tuple[Any, ...]:
        """Store one dataset(state, Q values) collected by teacher to buffer_path hard disk."""
        file_path = os.path.join(
            self.buffer_path,
            self.curr_time + "_transition_step_" + str(self.idx) + ".pkl",
        )
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.idx += 1

    def reset_dataloader(self):
        """Initialize and reset DataLoader class.
           DataLoader class must be reset for every epoch.
        """
        dataset = DistillationDataset(self.buffer_path, self.buffer_size)
        self.dataloader = iter(
            DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        )

    def sample_for_diltillation(self):
        """Sample a batch of state and Q-value for student's learning."""
        assert self.buffer_size >= self.batch_size

        return next(self.dataloader)


class DistillationDataset(Dataset):
    """Pytorch Dataset class for random batch data sampling.

    Attributes:
        buffer_path (str): distillation buffer path
        buffer_size (int): distillation buffer size

    """

    def __init__(self, buffer_path: str, buffer_size: int):
        """Initialize a DistillationBuffer object.

        Args:
            buffer_size (int): distillation buffer size
            buffer_path (str): distillation buffer path
            file_name_list (list): transition's file name list in distillation buffer path

        """
        self.buffer_path = buffer_path
        self.buffer_size = buffer_size
        self.file_name_list = os.listdir(self.buffer_path)

    def __len__(self):
        """Denotes the total number of samples."""
        return self.buffer_size

    def __getitem__(self, index):
        """Generates one sample of data."""
        file_path = os.path.join(self.buffer_path, self.file_name_list[index])
        with open(file_path, "rb") as f:
            transition = pickle.load(f)
        return transition
