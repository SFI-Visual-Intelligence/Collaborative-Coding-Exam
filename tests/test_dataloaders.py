from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import transforms

from CollaborativeCoding.dataloaders import (
    MNISTDataset0_3,
    MNISTDataset4_9,
    SVHNDataset,
    USPSDataset0_6,
    USPSH5_Digit_7_9_Dataset,
)
from CollaborativeCoding.load_data import load_data


@pytest.mark.parametrize(
    "data_name, expected",
    [
        ("usps_0-6", USPSDataset0_6),
        ("usps_7-9", USPSH5_Digit_7_9_Dataset),
        ("mnist_0-3", MNISTDataset0_3),
        ("svhn", SVHNDataset),
        ("mnist_4-9", MNISTDataset4_9),
    ],
)
def test_load_data(data_name, expected):
    print(data_name)
    dataset, _, _ = load_data(
        data_name,
        data_dir=Path("data"),
        transform=transforms.ToTensor(),
    )
    assert isinstance(dataset, expected)
    assert len(dataset) > 0
    assert isinstance(dataset[0], tuple)
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], int)
