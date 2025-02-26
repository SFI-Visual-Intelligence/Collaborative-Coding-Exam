from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class USPSH5_Digit_7_9_Dataset(Dataset):
    """
    Custom USPS dataset class that loads images with digits 7-9 from an .h5 file.

    Parameters
    ----------
    h5_path : str
        Path to the USPS `.h5` file.

    transform : callable, optional, default=None
        A transform function to apply on images. If None, no transformation is applied.

    Attributes
    ----------
    images : numpy.ndarray
        The filtered images corresponding to digits 7-9.

    labels : numpy.ndarray
        The filtered labels corresponding to digits 7-9.

    transform : callable, optional
        A transform function to apply to the images.
    """

    def __init__(
        self, data_path, sample_ids, train=False, transform=None, nr_channels=1
    ):
        super().__init__()
        """
        Initializes the USPS dataset by loading images and labels from the given `.h5` file.

        Parameters
        ----------
        h5_path : str
            Path to the USPS `.h5` file.
            
        transform : callable, optional, default=None
            A transform function to apply on images.
        """
        self.filename = "usps.h5"
        path = data_path if isinstance(data_path, Path) else Path(data_path)
        self.filepath = path / self.filename
        self.transform = transform
        self.mode = "train" if train else "test"
        self.h5_path = data_path / self.filename
        self.sample_ids = sample_ids
        self.nr_channels = nr_channels
        self.num_classes = 3

        # Load the dataset from the HDF5 file
        with h5py.File(self.filepath, "r") as hf:
            images = hf[self.mode]["data"][:]
            labels = hf[self.mode]["target"][:]

        # Filter only digits 7, 8, and 9
        mask = np.isin(labels, [7, 8, 9])
        self.images = images[mask]
        self.labels = labels[mask]
        # map labels from (7,9) to (0,2) for CE loss
        self.label_shift = lambda x: x - 7
        self.label_restore = lambda x: x + 7

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            The number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, id):
        """
        Returns a sample from the dataset given an index.

        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.

        Returns
        -------
        tuple
            - image (PIL Image): The image at the specified index.
            - label (int): The label corresponding to the image.
        """
        # Convert to PIL Image (USPS images are typically grayscale 16x16)
        image = Image.fromarray(self.images[id].astype(np.uint8), mode="L")
        label = int(self.labels[id])  # Convert label to integer
        label = self.label_shift(label)
        if self.transform:
            image = self.transform(image)

        return image, label

