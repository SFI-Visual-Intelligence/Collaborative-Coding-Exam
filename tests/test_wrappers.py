from pathlib import Path

import pytest
import torch as th

from CollaborativeCoding import MetricWrapper, load_data, load_model


def test_load_model():
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


def test_load_data():
    from torchvision import transforms

    dataset_names = [
        "usps_0-6",
        "mnist_0-3",
        "usps_7-9",
        "svhn",
        "mnist_4-9",
    ]

    trans = transforms.Compose(
        [
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ]
    )

    for name in dataset_names:
        dataset = load_data(name, train=False, data_dir=Path.cwd() / "Data", transform=trans)

        im, _ = dataset.__getitem__(0)

        assert dataset.__len__() != 0
        assert type(im) is th.Tensor and len(im.size()) == 3


def test_load_metric():
    metrics = ("entropy", "f1", "recall", "precision", "accuracy")

    class_sizes = [3, 6, 10]
    for class_size in class_sizes:
        y_true = th.rand((5, class_size)).argmax(dim=1)
        y_pred = th.rand((5, class_size))

        metricwrapper = MetricWrapper(
            *metrics,
            num_classes=class_size,
            macro_averaging=True if class_size % 2 == 0 else False,
        )

        metricwrapper(y_true, y_pred)
        metric = metricwrapper.getmetrics()
        assert metric is not None

        metricwrapper.resetmetric()
        metric2 = metricwrapper.getmetrics()
        assert metric != metric2
