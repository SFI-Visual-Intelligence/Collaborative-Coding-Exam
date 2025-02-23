from pathlib import Path

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
    dataset, _, _ = load_data(
        data_name,
        train=False,
        data_dir=Path("Data"),
        transform=transforms.ToTensor(),
    )

    sample = dataset[0]
    img, label = sample

    assert isinstance(dataset, expected), f"{type(dataset)} != {expected}"
    assert len(dataset) > 0, "Dataset is empty"
    assert isinstance(sample, tuple), f"{type(sample)} != tuple"
    assert isinstance(img, torch.Tensor), f"{type(img)} != torch.Tensor"
    assert isinstance(label, int), f"{type(label)} != int"
    assert len(img.size()) == 3, f"{len(img.size())} != 3"
