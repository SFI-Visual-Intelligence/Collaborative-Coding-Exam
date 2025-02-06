import pytest
import torch

from utils.models import ChristianModel, JanModel


@pytest.mark.parametrize(
    "image_shape, num_classes",
    [((1, 16, 16), 6), ((3, 16, 16), 6)],
)
def test_christian_model(image_shape, num_classes):
    n, c, h, w = 5, *image_shape

    model = ChristianModel(image_shape, num_classes)

    x = torch.randn(n, c, h, w)
    y = model(x)

    assert y.shape == (n, num_classes), f"Shape: {y.shape}"


@pytest.mark.parametrize(
    "image_shape, num_classes",
    [((1, 28, 28), 4), ((3, 16, 16), 10)],
)
def test_jan_model(image_shape, num_classes):
    n, c, h, w = 5, *image_shape

    model = JanModel(image_shape, num_classes)

    x = torch.randn(n, c, h, w)
    y = model(x)

    assert y.shape == (n, num_classes), f"Shape: {y.shape}"

