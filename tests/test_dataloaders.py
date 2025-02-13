from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from utils.dataloaders import MNISTDataset0_3, USPSDataset0_6, USPSH5_Digit_7_9_Dataset
from utils.load_data import load_data


@pytest.mark.parametrize(
    "data_name, expected",
    [
        ("usps_0-6", USPSDataset0_6),
        ("usps_7-9", USPSH5_Digit_7_9_Dataset),
        ("mnist_0-3", MNISTDataset0_3),
        # TODO: Add more datasets here
    ],
)
def test_load_data(data_name, expected):
    dataset = load_data(
        data_name,
        data_path=Path("data"),
        download=True,
        transform=transforms.ToTensor(),
    )
    assert isinstance(dataset, expected)
    assert len(dataset) > 0
    assert isinstance(dataset[0], tuple)
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(
        dataset[0][1], (int, torch.Tensor, np.ndarray)
    )  # Should probably restrict this to only int or one-hot encoded tensor or array for consistency.


def test_uspsdataset0_6():
    from tempfile import TemporaryDirectory

    import h5py
    import numpy as np
    from torchvision import transforms

    # Create a temporary directory (deleted after the test)
    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        tf = tempdir / "usps.h5"

        # Create a h5 file
        with h5py.File(tf, "w") as f:
            targets = np.array([6, 5, 4, 3, 2, 1, 0, 0, 0, 0])
            indices = np.arange(len(targets))
            # Populate the file with data
            f["train/data"] = np.random.rand(10, 16 * 16)
            f["train/target"] = targets

        trans = transforms.Compose(
            [
                transforms.Resize((16, 16)),
                transforms.ToTensor(),
            ]
        )
        dataset = USPSDataset0_6(
            data_path=tempdir,
            sample_ids=indices,
            train=True,
            transform=trans,
        )
        assert len(dataset) == 10
        data, target = dataset[0]
        assert data.shape == (1, 16, 16)
        assert target == 6
