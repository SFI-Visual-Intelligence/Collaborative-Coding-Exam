import numpy as np
from torch.utils.data import Dataset, random_split

from .dataloaders import (
    Downloader,
    MNISTDataset0_3,
    USPSDataset0_6,
    USPSH5_Digit_7_9_Dataset,
)


def filter_labels(samples: list, wanted_labels: list) -> list:
    return list(filter(lambda x: x in wanted_labels, samples))


def load_data(dataset: str, *args, **kwargs) -> tuple:
    """
    load the dataset based on the dataset name.

    Args
    ----
    dataset : str
        Name of the dataset to load.
    *args : list
        Additional arguments for the dataset class.
    **kwargs : dict
        Additional keyword arguments for the dataset class.

    Returns
    -------
    tuple
        Tuple of train, validation and test datasets.

    Raises
    ------
    NotImplementedError
        If the dataset is not implemented.

    Examples
    --------
    >>> from utils import setup_data
    >>> train, val, test = setup_data("usps_0-6", data_path="data", train=True, download=True)
    >>> len(train), len(val), len(test)
    (4914, 546, 1782)
    """

    match dataset.lower():
        case "usps_0-6":
            dataset = USPSDataset0_6
            train_labels, test_labels = Downloader.usps(*args, **kwargs)
            labels = np.arange(7)
        case "usps_7-9":
            dataset = USPSH5_Digit_7_9_Dataset
            train_labels, test_labels = Downloader.usps(*args, **kwargs)
            labels = np.arange(7, 10)
        case "mnist_0-3":
            dataset = MNISTDataset0_3
            train_labels, test_labels = Downloader.mnist(*args, **kwargs)
            labels = np.arange(4)
        case _:
            raise NotImplementedError(f"Dataset: {dataset} not implemented.")

    val_size = kwargs.get("val_size", 0.2)

    # Get the indices of the samples
    train_indices = np.arange(len(train_labels))
    test_indices = np.arange(len(test_labels))

    # Filter the labels to only get indices of the wanted labels
    train_samples = train_indices[np.isin(train_labels, labels)]
    test_samples = test_indices[np.isin(test_labels, labels)]

    train_samples, val_samples = random_split(train_samples, [1 - val_size, val_size])

    train = dataset(
        *args,
        sample_ids=train_samples,
        train=True,
        **kwargs,
    )

    val = dataset(
        *args,
        sample_ids=val_samples,
        train=True,
        **kwargs,
    )

    test = dataset(
        *args,
        sample_ids=test_samples,
        train=False,
        **kwargs,
    )

    return train, val, test
