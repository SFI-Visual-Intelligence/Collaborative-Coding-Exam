"""
Dataset class for USPS dataset with labels 0-6.

This module contains the Dataset class for the USPS dataset with labels 0-6.
"""

from pathlib import Path

import h5py as h5
import numpy as np
from torch.utils.data import Dataset


class USPSDataset0_6(Dataset):
    """
    Dataset class for USPS dataset with labels 0-6.

    Args
    ----
    data_path : pathlib.Path
        Path to the USPS dataset file.
    train : bool, optional
        Mode of the dataset.
    transform : callable, optional
        A function/transform that takes in a sample and returns a transformed version.
    download : bool, optional
        Whether to download the Dataset.

    Attributes
    ----------
    path : pathlib.Path
        Path to the USPS dataset file.
    mode : str
        Mode of the dataset, either train or test.
    transform : callable
        A function/transform that takes in a sample and returns a transformed version.
    idx : numpy.ndarray
        Indices of samples with labels 0-6.
    num_classes : int
        Number of classes in the dataset

    Methods
    -------
    _index()
        Get indices of samples with labels 0-6.
    _load_data(idx)
        Load data and target label from the dataset.
    __len__()
        Get the number of samples in the dataset.
    __getitem__(idx)
        Get a sample from the dataset.

    Examples
    --------
    >>> from src.datahandlers import USPSDataset0_6
    >>> dataset = USPSDataset0_6(path="data/usps.h5", mode="train")
    >>> len(dataset)
    5460
    >>> data, target = dataset[0]
    >>> data.shape
    (16, 16)
    >>> target
    6
    """

    def __init__(
        self,
        data_path: Path,
        train: bool = False,
        transform=None,
        download: bool = False,
    ):
        super().__init__()
        self.path = data_path
        self.transform = transform
        self.num_classes = 7

        if download:
            raise NotImplementedError("Download functionality not implemented.")

        self.mode = "train" if train else "test"
        self.idx = self._index()

    def _index(self):
        with h5.File(self.path, "r") as f:
            labels = f[self.mode]["target"][:]

        # Get indices of samples with labels 0-6
        mask = labels <= 6
        idx = np.where(mask)[0]

        return idx

    def _load_data(self, idx):
        with h5.File(self.path, "r") as f:
            data = f[self.mode]["data"][idx]
            label = f[self.mode]["target"][idx]

        return data, label

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        data, target = self._load_data(self.idx[idx])

        data = data.reshape(16, 16)

        # one hot encode the target
        target = np.eye(self.num_classes, dtype=np.float32)[target]

        # Add channel dimension
        data = np.expand_dims(data, axis=0)

        if self.transform:
            data = self.transform(data)

        return data, target
