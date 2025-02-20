from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torchvision import transforms

from CollaborativeCoding import load_data, load_metric, load_model
from CollaborativeCoding.dataloaders import (
    MNISTDataset0_3,
    SVHNDataset,
    USPSDataset0_6,
    USPSH5_Digit_7_9_Dataset,
)


def test_load_model():
    import torch as th

    image_shape = (1, 16, 16)
    num_classes = 4

    dummy_img = th.rand((1, *image_shape))

    modelnames = [
        "magnusmodel",
        "christianmodel",
        "janmodel",
        "solveigmodel",
        "johanmodel",
    ]

    for name in modelnames:
        print(name)
        model = load_model(name, image_shape=image_shape, num_classes=num_classes)

        with th.no_grad():
            output = model(dummy_img)
            assert output.size() == (1, 4), (
                f"Model {name} returned image of size {output}. Expected (1,4)"
            )


@pytest.mark.parametrize(
    "data_name, expected",
    [
        ("usps_0-6", USPSDataset0_6),
        ("usps_7-9", USPSH5_Digit_7_9_Dataset),
        ("mnist_0-3", MNISTDataset0_3),
        ("svhn", SVHNDataset),
    ],
)
def test_load_data(data_name, expected):
    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        train, val, test = load_data(
            data_name,
            data_dir=tempdir,
            transform=transforms.ToTensor(),
        )

        for dataset in [train, val, test]:
            assert isinstance(dataset, expected)
            assert len(dataset) > 0
            assert isinstance(dataset[0], tuple)
            assert isinstance(dataset[0][0], torch.Tensor)
            assert isinstance(dataset[0][1], int)


def test_load_metric():
    pass
