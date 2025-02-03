from torch.utils.data import Dataset

from .dataloaders import USPSDataset0_6, USPSH5_Digit_7_9_Dataset


def load_data(dataset: str, *args, **kwargs) -> Dataset:
    match dataset.lower():
        case "usps_0-6":
            return USPSDataset0_6(*args, **kwargs)
        case "usps_7-9":
            return  USPSH5_Digit_7_9_Dataset(*args, **kwargs)
        case _:
            raise ValueError(f"Dataset: {dataset} not implemented.")
