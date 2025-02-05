import gzip
import os
import urllib.request
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class MNISTDataset0_3(Dataset):
    """
    A custom dataset class for loading MNIST data, specifically for digits 0 through 3.
    Parameters
    ----------
    data_path : Path
        The root directory where the MNIST data is stored or will be downloaded.
    train : bool, optional
        If True, loads the training data, otherwise loads the test data. Default is False.
    transform : callable, optional
        A function/transform that takes in an image and returns a transformed version. Default is None.
    download : bool, optional
        If True, downloads the dataset if it is not already present in the specified data_path. Default is False.
    Attributes
    ----------
    data_path : Path
        The root directory where the MNIST data is stored.
    mnist_path : Path
        The directory where the MNIST data files are stored.
    train : bool
        Indicates whether the training data or test data is being used.
    transform : callable
        A function/transform that takes in an image and returns a transformed version.
    download : bool
        Indicates whether the dataset should be downloaded if not present.
    images_path : Path
        The path to the image file (training or test) based on the `train` flag.
    labels_path : Path
        The path to the label file (training or test) based on the `train` flag.
    idx : numpy.ndarray
        Indices of the labels that are less than 4.
    length : int
        The number of samples in the dataset.
    Methods
    -------
    _parse_labels(train)
        Parses the labels from the label file.
    _chech_is_downloaded()
        Checks if the dataset is already downloaded.
    _download_data()
        Downloads and extracts the MNIST dataset.
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(index)
        Returns the image and label at the specified index.
    """

    def __init__(
        self,
        data_path: Path,
        train: bool = False,
        transform=None,
        download: bool = False,
    ):
        super().__init__()

        self.data_path = data_path
        self.mnist_path = self.data_path / "MNIST"
        self.train = train
        self.transform = transform
        self.download = download
        self.num_classes = 4

        if not self.download and not self._chech_is_downloaded():
            raise ValueError(
                "Data not found. Set --download-data=True to download the data."
            )
        if self.download and not self._chech_is_downloaded():
            self._download_data()

        self.images_path = self.mnist_path / (
            "train-images-idx3-ubyte" if train else "t10k-images-idx3-ubyte"
        )
        self.labels_path = self.mnist_path / (
            "train-labels-idx1-ubyte" if train else "t10k-labels-idx1-ubyte"
        )

        labels = self._parse_labels(train=self.train)

        self.idx = np.where(labels < 4)[0]

        self.length = len(self.idx)

    def _parse_labels(self, train):
        with open(self.labels_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data

    def _chech_is_downloaded(self):
        if self.mnist_path.exists():
            required_files = [
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte",
            ]
            if all([(self.mnist_path / file).exists() for file in required_files]):
                print("MNIST Dataset already downloaded.")
                return True
            else:
                return False
        else:
            self.mnist_path.mkdir(parents=True, exist_ok=True)
            return False

    def _download_data(self):
        urls = {
            "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        }

        for name, url in urls.items():
            file_path = os.path.join(self.mnist_path, url.split("/")[-1])
            if not os.path.exists(file_path.replace(".gz", "")):  # Avoid re-downloading
                urllib.request.urlretrieve(url, file_path)
                with gzip.open(file_path, "rb") as f_in:
                    with open(file_path.replace(".gz", ""), "wb") as f_out:
                        f_out.write(f_in.read())
                os.remove(file_path)  # Remove compressed file

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with open(self.labels_path, "rb") as f:
            f.seek(8 + self.idx[index])  # Jump to the label position
            label = int.from_bytes(f.read(1), byteorder="big")  # Read 1 byte for label

        with open(self.images_path, "rb") as f:
            f.seek(16 + self.idx[index] * 28 * 28)  # Jump to image position
            image = np.frombuffer(f.read(28 * 28), dtype=np.uint8).reshape(
                28, 28
            )  # Read image data

        image = np.expand_dims(image, axis=0)  # Add channel dimension

        if self.transform:
            image = self.transform(image)

        return image, label
    

