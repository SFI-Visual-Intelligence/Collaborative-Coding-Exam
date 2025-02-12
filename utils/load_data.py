from torch.utils.data import Dataset

from .dataloaders import (
    MNISTDataset0_3,
    SVHNDataset,
    USPSDataset0_6,
    USPSH5_Digit_7_9_Dataset,
)


def load_data(dataset: str, *args, **kwargs) -> Dataset:
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
    dataset : torch.utils.data.Dataset
        Dataset object.

    Raises
    ------
    NotImplementedError
        If the dataset is not implemented.

    Examples
    --------
    >>> from utils import load_data
    >>> dataset = load_data("usps_0-6", data_path="data", train=True, download=True)
    >>> len(dataset)
    5460
    """
    match dataset.lower():
        case "usps_0-6":
            return USPSDataset0_6(*args, **kwargs)
        case "mnist_0-3":
            return MNISTDataset0_3(*args, **kwargs)
        case "usps_7-9":
            return USPSH5_Digit_7_9_Dataset(*args, **kwargs)
        case "svhn":
            return SVHNDataset(*args, **kwargs)
        case "mnist_4-9":
            raise NotImplementedError("MNIST 4-9 dataset not yet implemented.")
        case _:
            raise NotImplementedError(f"Dataset: {dataset} not implemented.")
