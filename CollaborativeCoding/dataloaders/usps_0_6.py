"""
Dataset class for USPS dataset with labels 0-6.

This module contains the Dataset class for the USPS dataset with labels 0-6.
"""

from pathlib import Path

import h5py as h5
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class USPSDataset0_6(Dataset):
    """
    Dataset class for USPS dataset with labels 0-6.

    Args
    ----
    data_path : pathlib.Path
        Path to the data directory.
    train : bool, optional
        Mode of the dataset.
    transform : callable, optional
        A function/transform that takes in a sample and returns a transformed version.
    download : bool, optional
        Whether to download the Dataset.

    Attributes
    ----------
    filepath : pathlib.Path
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
    >>> from torchvision import transforms
    >>> from src.datahandlers import USPSDataset0_6
    >>> transform = transforms.Compose([
    ...     transforms.Resize((16, 16)),
    ...     transforms.ToTensor()
    ... ])
    >>> dataset = USPSDataset0_6(
    ...     data_path="data",
    ...     transform=transform
    ...     download=True,
    ...     train=True,
    ... )
    >>> len(dataset)
    5460
    >>> data, target = dataset[0]
    >>> data.shape
    (1, 16, 16)
    >>> target
    tensor([1., 0., 0., 0., 0., 0., 0.])
    """

    filename = "usps.h5"
    num_classes = 7

    def __init__(
        self,
        data_path: Path,
        sample_ids: list,
        train: bool = False,
        transform=None,
        nr_channels=1,
    ):
        super().__init__()

        path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.filepath = path / self.filename
        self.transform = transform
        self.mode = "train" if train else "test"
        self.sample_ids = sample_ids
        self.nr_channels = nr_channels

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, id):
        index = self.sample_ids[id]

        with h5.File(self.filepath, "r") as f:
            data = f[self.mode]["data"][index].astype(np.uint8)
            label = int(f[self.mode]["target"][index])

        if self.nr_channels == 1:
            data = Image.fromarray(data, mode="L")
        elif self.nr_channels == 3:
            data = Image.fromarray(data, mode="RGB")
        else:
            raise ValueError("Invalid number of channels")

        if self.transform:
            data = self.transform(data)

        # label = torch.tensor(label).long()

        return data, label
