from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class USPSH5_Digit_7_9_Dataset(Dataset):
    """
    This class loads a subset of the USPS dataset, specifically images of digits 7, 8, and 9, from an HDF5 file.
    It allows for applying transformations to the images and provides methods to retrieve images and their corresponding labels.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing the USPS `.h5` file. This file should contain the data in the "train" or "test" group.

    sample_ids : list of int
        A list of sample indices to be used from the dataset. This allows for filtering or selecting a subset of the full dataset.

    train : bool, optional, default=False
        If `True`, the dataset is loaded in training mode (using the "train" group). If `False`, the dataset is loaded in test mode (using the "test" group).

    transform : callable, optional, default=None
        A transformation function to apply to each image. If `None`, no transformation is applied. Typically used for data augmentation or normalization.

    nr_channels : int, optional, default=1
        The number of channels in the image. USPS images are typically grayscale, so this should generally be set to 1. This parameter allows for potential future flexibility.

    Attributes
    ----------
    images : numpy.ndarray
        Array of images corresponding to digits 7, 8, and 9 from the USPS dataset. The images are loaded from the HDF5 file and filtered based on the labels.

    labels : numpy.ndarray
        Array of labels corresponding to the images. Only labels of digits 7, 8, and 9 are retained, and they are mapped to 0, 1, and 2 for classification tasks.

    transform : callable, optional
        A transformation function to apply to the images. This is passed as an argument during initialization.

    label_shift : function
        A function to shift the labels for classification purposes. It maps the original labels (7, 8, 9) to (0, 1, 2).

    label_restore : function
        A function to restore the original labels (7, 8, 9) from the shifted labels (0, 1, 2).

    num_classes : int
        The number of unique labels in the dataset, which is 3 (for digits 7, 8, and 9).
    """

    def __init__(
        self, data_path, sample_ids, train=False, transform=None, nr_channels=1
    ):
        super().__init__()
        """
        Initializes the USPS dataset by loading images and labels from the given `.h5` file.
        
        The dataset is filtered to only include images of digits 7, 8, and 9, which are mapped to labels 0, 1, and 2 respectively for classification purposes.
        
        Parameters
        ----------
        data_path : str or Path
          Path to the directory containing the USPS `.h5` file.
        
        sample_ids : list of int
          List of sample indices to load from the dataset.
        
        train : bool, optional, default=False
          If `True`, loads the training data from the HDF5 file. If `False`, loads the test data.
        
        transform : callable, optional, default=None
          A function to apply transformations to the images. If None, no transformation is applied.
        
        nr_channels : int, optional, default=1
          The number of channels in the image. Defaults to 1 for grayscale images.
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

        This method is required for PyTorch's Dataset class, as it allows PyTorch to determine the size of the dataset.

        Returns
        -------
        int
            The number of images in the dataset (after filtering for digits 7, 8, and 9).
        """

        return len(self.images)

    def __getitem__(self, id):
        """
        Returns a sample from the dataset given an index.

        This method is required for PyTorch's Dataset class, as it allows indexing into the dataset to retrieve specific samples.

        Parameters
        ----------
        idx : int
           The index of the sample to retrieve from the dataset.

        Returns
        -------
        tuple
           A tuple containing:
           - image (PIL Image): The image at the specified index.
           - label (int): The label corresponding to the image, shifted to be in the range [0, 2] for classification.
        """
        # Convert to PIL Image (USPS images are typically grayscale 16x16)
        image = Image.fromarray(self.images[id].astype(np.uint8), mode="L")
        label = int(self.labels[id])  # Convert label to integer
        label = self.label_shift(label)
        if self.transform:
            image = self.transform(image)

        return image, label
