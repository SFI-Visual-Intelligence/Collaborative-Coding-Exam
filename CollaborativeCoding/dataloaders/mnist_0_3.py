from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .datasources import MNIST_SOURCE


class MNISTDataset0_3(Dataset):
    """
    A custom Dataset class for loading a subset of the MNIST dataset containing digits 0 to 3.

    Args
    ----------
    data_path : Path
        The root directory where the MNIST folder with data is stored.
    sample_ids : list
        A list of indices specifying which samples to load.
    train : bool, optional
        If True, load training data, otherwise load test data. Default is False.
    transform : callable, optional
        A function/transform to apply to the images. Default is None.

    Attributes
    ----------
    mnist_path : Path
        The directory where the MNIST dataset is located within the root directory.
    idx : list
        A list of indices specifying which samples to load.
    train : bool
        Indicates whether to load training data or test data.
    transform : callable
        A function/transform to apply to the images.
    num_classes : int
        The number of classes in the dataset (0 to 3).
    images_path : Path
        The path to the image file (train or test) based on the `train` flag.
    labels_path : Path
        The path to the label file (train or test) based on the `train` flag.
    length : int
        The number of samples in the dataset.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(index)
        Retrieves the image and label at the specified index.
    """

    def __init__(
        self,
        data_path: Path,
        sample_ids: list,
        train: bool = False,
        transform=None,
        nr_channels: int = 1,
    ):
        super().__init__()

        self.mnist_path = data_path / "MNIST"
        self.idx = sample_ids
        self.train = train
        self.transform = transform
        self.num_classes = 4

        self.images_path = self.mnist_path / (
            MNIST_SOURCE["train_images"][1] if train else MNIST_SOURCE["test_images"][1]
        )
        self.labels_path = self.mnist_path / (
            MNIST_SOURCE["train_labels"][1] if train else MNIST_SOURCE["test_labels"][1]
        )

        self.length = len(self.idx)

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

        # image = np.expand_dims(image, axis=0)  # Add channel dimension
        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)
        return image, label
