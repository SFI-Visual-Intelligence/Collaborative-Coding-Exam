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
            train_samples, test_samples = Downloader.usps(*args, **kwargs)
            labels = range(7)
        case "usps_7-9":
            dataset = USPSH5_Digit_7_9_Dataset
            train_samples, test_samples = Downloader.usps(*args, **kwargs)
            labels = range(7, 10)
        case "mnist_0-3":
            dataset = MNISTDataset0_3
            train_samples, test_samples = Downloader.mnist(*args, **kwargs)
            labels = range(4)
        case _:
            raise NotImplementedError(f"Dataset: {dataset} not implemented.")

    val_size = kwargs.get("val_size", 0.1)

    train_samples = filter_labels(train_samples, labels)
    test_samples = filter_labels(test_samples, labels)

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
