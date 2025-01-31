from dataloaders import USPS_0_6
from torch.utils.data import Dataset


def load_data(dataset: str) -> Dataset:
    match dataset.lower():
        case "usps_0-6":
            return USPS_0_6
        case _:
            raise ValueError(f"Dataset: {dataset} not implemented.")
