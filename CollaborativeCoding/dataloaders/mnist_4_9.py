from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .datasources import MNIST_SOURCE


class MNISTDataset4_9(Dataset):
    """
    MNIST dataset of numbers 4-9.

    Parameters
    ----------
    data_path : Path
        Root directory where MNIST dataset is stored
    sample_ids : np.ndarray
        Array of indices spcifying which samples to load. This determines the samples used by the dataloader.
    train : bool, optional
        Whether to train the model or not, by default False
    transorm : callable, optional
        Transform to apply to the images, by default None
    nr_channels : int, optional
        Number of channels in the images, by default 1
    """

    def __init__(
        self,
        data_path: Path,
        sample_ids: np.ndarray,
        train: bool = False,
        transform=None,
        nr_channels: int = 1,
    ):
        super().__init__()
        self.data_path = data_path
        self.mnist_path = self.data_path / "MNIST"
        self.samples = sample_ids
        self.train = train
        self.transform = transform
        self.num_classes = 6

        self.images_path = self.mnist_path / (
            MNIST_SOURCE["train_images"][1] if train else MNIST_SOURCE["test_images"][1]
        )
        self.labels_path = self.mnist_path / (
            MNIST_SOURCE["train_labels"][1] if train else MNIST_SOURCE["test_labels"][1]
        )

        # Functions to map the labels from (4,9) -> (0,5) for CrossEntropyLoss to work properly.
        self.label_shift = lambda x: x - 4
        self.label_restore = lambda x: x + 4

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        with open(self.labels_path, "rb") as labelfile:
            label_pos = 8 + self.samples[idx]
            labelfile.seek(label_pos)
            label = int.from_bytes(labelfile.read(1), byteorder="big")

        with open(self.images_path, "rb") as imagefile:
            image_pos = 16 + self.samples[idx] * 28 * 28
            imagefile.seek(image_pos)
            image = np.frombuffer(imagefile.read(28 * 28), dtype=np.uint8).reshape(
                28, 28
            )

        # image = np.expand_dims(image, axis=0)  # Channel
        image = Image.fromarray(image.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, self.label_shift(label)
