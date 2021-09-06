# -*- coding: utf-8 -*-
"""Distillation buffer."""

import os
import pickle
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DistillationBuffer:
    """Class for managing reading and writing of distillation data.
       Distillation data is stored in the dataset_path location on the actual hard disk.
       The data collected by the teacher is stored as individual pickle files.
       It is also read as batch through the pytorch DataLoader class.

    Attributes:
        batch_size (int): size of batch size from distillation buffer for training
        dataset_path (list): list of distillation buffer path
        curr_time (str): program's start time to distinguish between teacher agents
        idx (int): index of data
        buffer_size (int): distillation buffer size
        dataloader (DataLoader): pytorch library for random batch data sampling

    """

    def __init__(
        self,
        batch_size: int,
        dataset_path: List[str],
    ):
        """Initialize a DistillationBuffer object.

        Args:
            batch_size (int): size of a batched sampled from distillation buffer for training
            dataset_path (list): list of distillation buffer path
            curr_time (str): program's start time to distinguish between teacher agents

        """
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.idx = 0
        self.buffer_size = 0
        self.dataloader = None
        self.is_contain_q = False

    def reset_dataloader(self):
        """Initialize and reset DataLoader class.
        DataLoader class must be reset for every epoch.
        """
        dataset = DistillationDataset(self.dataset_path)
        self.is_contain_q = dataset.is_contain_q
        self.buffer_size = len(dataset)
        self.dataloader = iter(
            DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        )

    def sample_for_diltillation(self):
        """Sample a batch of state and Q-value for student's learning."""
        assert (
            self.buffer_size >= self.batch_size
        ), f"buffer size({self.buffer_size}) < ({self.batch_size})"

        return next(self.dataloader)


class DistillationDataset(Dataset):
    """Pytorch Dataset class for random batch data sampling.

    Attributes:
        dataset_path (str): distillation buffer path

    """

    def __init__(self, dataset_path: List[str]):
        """Initialize a DistillationBuffer object.

        Args:
            dataset_path (str): distillation buffer path
            file_name_list (list): transition's file name list in distillation buffer path

        """
        super().__init__()
        self.dataset_path = dataset_path
        self.file_name_list = []

        sum_data_len = 0
        for _dir in self.dataset_path:
            tmp = os.listdir(_dir)
            self.file_name_list += [os.path.join(_dir, x) for x in tmp]
            with open(self.file_name_list[-1], "rb") as f:
                data = pickle.load(f)
            sum_data_len += int(len(data) == 2)

        if sum_data_len == len(self.dataset_path):
            self.is_contain_q = True
        elif sum_data_len == 0:
            self.is_contain_q = False
        else:
            raise AssertionError(
                "There is a mixture of data with q present and non-existent ones"
                + "in buffer-path."
            )

    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.file_name_list)

    def __getitem__(self, index):
        """Generates one sample of data."""
        with open(self.file_name_list[index], "rb") as f:
            transition = pickle.load(f)
        return transition
