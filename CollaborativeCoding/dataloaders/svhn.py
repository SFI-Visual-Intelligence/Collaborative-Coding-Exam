import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets import SVHN


class SVHNDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        sample_ids: list,
        train: bool,
        transform=None,
        nr_channels=3,
    ):
        """
        Initializes the SVHNDataset object for loading the Street View House Numbers (SVHN) dataset.
        Args:
            data_path (str): Path to where the data is stored. If `download` is set to True, this is where the data will be downloaded.
            train (bool): If True, loads the training split of the dataset; otherwise, loads the test split.
            transform (callable, optional): A function/transform to apply to the images.
            download (bool): If True, downloads the dataset to the specified `data_path`.
            nr_channels (int): Number of channels in the images. Default is 3 for RGB images.
        Raises:
            AssertionError: If the split is not 'train' or 'test'.
        """
        super().__init__()

        self.data_path = data_path / "SVHN"
        self.indexes = sample_ids
        self.split = "train" if train else "test"

        self.nr_channels = nr_channels
        self.transforms = transform

        if not os.path.exists(
            os.path.join(self.data_path, f"svhn_{self.split}data.h5")
        ):
            self._create_h5py(self.data_path)

        assert os.path.exists(
            os.path.join(self.data_path, f"svhn_{self.split}data.h5")
        ), f"File svhn_{self.split}data.h5 does not exists. Run download=True"
        with h5py.File(
            os.path.join(self.data_path, f"svhn_{self.split}data.h5"), "r"
        ) as h5f:
            self.labels = h5f["labels"][:]

        self.num_classes = len(np.unique(self.labels))

    def _create_h5py(self, path: str):
        """
        Downloads the SVHN dataset to the specified directory.
        Args:
            path (str): The directory where the dataset will be downloaded.
        """
        print(f"Downloading SVHN data into {path}")

        data = loadmat(os.path.join(path, f"{self.split}_32x32.mat"))

        images, labels = data["X"], data["y"]
        images = images.transpose(3, 1, 0, 2)
        labels[labels == 10] = 0
        labels = labels.flatten()

        with h5py.File(
            os.path.join(self.data_path, f"svhn_{self.split}data.h5"), "w"
        ) as h5f:
            h5f.create_dataset("images", data=images)
            h5f.create_dataset("labels", data=labels)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: The number of samples.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Retrieves the image and label at the specified index.
        Args:
            index (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        lab = self.labels[index]
        with h5py.File(
            os.path.join(self.data_path, f"svhn_{self.split}data.h5"), "r"
        ) as h5f:
            img = Image.fromarray(h5f["images"][index])

        if self.nr_channels == 1:
            img = img.convert("L")
        if self.transforms is not None:
            img = self.transforms(img)

        return img, int(lab)
