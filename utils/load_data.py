from torch.utils.data import Dataset

from .dataloaders import USPSDataset0_6, MNISTDataset0_3


def load_data(dataset: str, *args, **kwargs) -> Dataset:
    match dataset.lower():
        case "usps_0-6":
            return USPSDataset0_6(*args, **kwargs)
        case "mnist_0-3":
            return MNISTDataset0_3(*args, **kwargs)
        case _:
            raise ValueError(f"Dataset: {dataset} not implemented.")
