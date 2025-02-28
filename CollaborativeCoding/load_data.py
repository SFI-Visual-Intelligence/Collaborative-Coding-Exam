import numpy as np
from torch.utils.data import random_split

from .dataloaders import (
    Downloader,
    MNISTDataset0_3,
    MNISTDataset4_9,
    SVHNDataset,
    USPSDataset0_6,
    USPSH5_Digit_7_9_Dataset,
)


def filter_labels(samples: list, wanted_labels: list) -> list:
    return list(filter(lambda x: x in wanted_labels, samples))


def load_data(dataset: str, *args, **kwargs) -> tuple:
    """
    Load the dataset based on the dataset name.

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
    >>> from CollaborativeCoding import setup_data
    >>> train, val, test = setup_data("usps_0-6", data_path="data", train=True, download=True)
    >>> len(train), len(val), len(test)
    (4914, 546, 1782)
    """
    downloader = Downloader()
    data_dir = kwargs.get("data_dir")
    transform = kwargs.get("transform")
    match dataset.lower():
        case "usps_0-6":
            dataset = USPSDataset0_6
            train_labels, test_labels = downloader.usps(data_dir=data_dir)
            labels = np.arange(7)
        case "usps_7-9":
            dataset = USPSH5_Digit_7_9_Dataset
            train_labels, test_labels = downloader.usps(data_dir=data_dir)
            labels = np.arange(7, 10)
        case "mnist_0-3":
            dataset = MNISTDataset0_3
            train_labels, test_labels = downloader.mnist(data_dir=data_dir)
            labels = np.arange(4)
        case "svhn":
            dataset = SVHNDataset
            train_labels, test_labels = downloader.svhn(data_dir=data_dir)
            labels = np.unique(train_labels)
        case "mnist_4-9":
            dataset = MNISTDataset4_9
            train_labels, test_labels = downloader.mnist(data_dir=data_dir)
            labels = np.arange(4, 10)
        case _:
            raise NotImplementedError(f"Dataset: {dataset} not implemented.")

    val_size = kwargs.get("val_size", 0.2)

    # Get the indices of the samples
    train_indices = np.arange(len(train_labels))
    test_indices = np.arange(len(test_labels))

    # Filter the labels to only get indices of the wanted labels
    train_samples = train_indices[np.isin(train_labels, labels).flatten()]
    test_samples = test_indices[np.isin(test_labels, labels).flatten()]

    train_samples, val_samples = random_split(train_samples, [1 - val_size, val_size])

    train = dataset(
        data_path=data_dir,
        sample_ids=train_samples,
        train=True,
        transform=transform,
        nr_channels=kwargs.get("nr_channels", 1),
    )

    val = dataset(
        data_path=data_dir,
        sample_ids=val_samples,
        train=True,
        transform=transform,
        nr_channels=kwargs.get("nr_channels", 1),
    )

    test = dataset(
        data_path=data_dir,
        sample_ids=test_samples,
        train=False,
        transform=transform,
        nr_channels=kwargs.get("nr_channels", 1),
    )

    return train, val, test
