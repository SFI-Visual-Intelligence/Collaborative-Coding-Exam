import pytest
import torch

from CollaborativeCoding import load_model


@pytest.mark.parametrize(
    "model_name",
    [
        "magnusmodel",
        "christianmodel",
        "janmodel",
        "johanmodel",
        "solveigmodel",
    ],
)
@pytest.mark.parametrize("image_shape", [(i, 28, 28) for i in [1, 3]])
@pytest.mark.parametrize("num_classes", [3, 6, 10])
def test_load_model(model_name, image_shape, num_classes):
    model = load_model(model_name, image_shape, num_classes)

    n, c, h, w = 5, *image_shape

    dummy_img = torch.randn(n, c, h, w)
    with torch.no_grad():
        y = model(dummy_img)

    assert y.shape == (n, num_classes), f"Shape: {y.shape} != {(n, num_classes)}"
