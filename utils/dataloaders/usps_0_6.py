"""
Dataset class for USPS dataset with labels 0-6.

This module contains the Dataset class for the USPS dataset with labels 0-6.
"""

import bz2
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from urllib.request import urlretrieve

import h5py as h5
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .datasources import USPS_SOURCE


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

    filename = "usps.h5"

    def __init__(
        self,
        data_path: Path,
        train: bool = False,
        transform=None,
        download: bool = False,
    ):
        super().__init__()

        path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.filepath = path / self.filename
        self.transform = transform
        self.num_classes = 7  # 0-6
        self.mode = "train" if train else "test"

        # Download the dataset if it does not exist in a temporary directory
        # to automatically clean up the downloaded file
        if download and not self._dataset_ok():
            url, _, checksum = USPS_SOURCE[self.mode]

            print(f"Downloading USPS dataset ({self.mode})...")
            self.download(url, self.filepath, checksum, self.mode)

        self.idx = self._index()

    def _dataset_ok(self):
        """Check if the dataset file exists and contains the required datasets."""

        if not self.filepath.exists():
            print(f"Dataset file {self.filepath} does not exist.")
            return False

        with h5.File(self.filepath, "r") as f:
            for mode in ["train", "test"]:
                if mode not in f:
                    print(
                        f"Dataset file {self.filepath} is missing the {mode} dataset."
                    )
                    return False

        return True

    def download(self, url, filepath, checksum, mode):
        """Download the USPS dataset."""

        def reporthook(blocknum, blocksize, totalsize):
            """Report download progress."""
            denom = 1024 * 1024
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 1e2 / totalsize
                s = f"\r{int(percent):^3}% {readsofar / denom:.2f} of {totalsize / denom:.2f} MB"
                print(s, end="", flush=True)
                if readsofar >= totalsize:
                    print()

        # Download the dataset to a temporary file
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tmpfile = tmpdir / "usps.bz2"
            urlretrieve(
                url,
                tmpfile,
                reporthook=reporthook,
            )

            # For fun we can check the integrity of the downloaded file
            if not self.check_integrity(tmpfile, checksum):
                errmsg = (
                    "The checksum of the downloaded file does "
                    "not match the expected checksum."
                )
                raise ValueError(errmsg)

            # Load the dataset and save it as an HDF5 file
            with bz2.open(tmpfile) as fp:
                raw = [line.decode().split() for line in fp.readlines()]

                tmp = [[x.split(":")[-1] for x in data[1:]] for data in raw]

                imgs = np.asarray(tmp, dtype=np.float32)
                imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)

                targets = [int(d[0]) - 1 for d in raw]

        with h5.File(self.filepath, "a") as f:
            f.create_dataset(f"{mode}/data", data=imgs, dtype=np.float32)
            f.create_dataset(f"{mode}/target", data=targets, dtype=np.int32)

    @staticmethod
    def check_integrity(filepath, checksum):
        """Check the integrity of the USPS dataset file."""

        file_hash = hashlib.md5(filepath.read_bytes()).hexdigest()

        return checksum == file_hash

    def _index(self):
        with h5.File(self.filepath, "r") as f:
            labels = f[self.mode]["target"][:]

        # Get indices of samples with labels 0-6
        mask = labels <= 6
        idx = np.where(mask)[0]

        return idx

    def _load_data(self, idx):
        with h5.File(self.filepath, "r") as f:
            data = f[self.mode]["data"][idx].astype(np.uint8)
            label = f[self.mode]["target"][idx]

        return data, label

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        data, target = self._load_data(self.idx[idx])
        data = Image.fromarray(data, mode="L")

        # one hot encode the target
        target = np.eye(self.num_classes, dtype=np.float32)[target]

        if self.transform:
            data = self.transform(data)

        return data, target


if __name__ == "__main__":
    # Example usage:
    transform = transforms.Compose(
        [
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ]
    )

    dataset = USPSDataset0_6(
        data_path="data",
        train=True,
        download=False,
        transform=transform,
    )
    print(len(dataset))
    data, target = dataset[0]
    print(data.shape)
    print(target)
