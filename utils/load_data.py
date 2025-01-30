from torch.utils.data import Dataset


def load_data(dataset: str) -> Dataset:
    raise ValueError(
        f"Dataset: {dataset} not implemented. \nCheck the documentation for implemented metrics, or check your spelling"
    )
