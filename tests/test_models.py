import pytest
import torch

from utils.models import ChristianModel


@pytest.mark.parametrize("in_channels, num_classes", [(1, 6), (3, 6)])
def test_christian_model(in_channels, num_classes):
    n, c, h, w = 5, in_channels, 16, 16

    model = ChristianModel(c, num_classes)

    x = torch.randn(n, c, h, w)
    y = model(x)

    assert y.shape == (n, num_classes), f"Shape: {y.shape}"
    assert y.sum(dim=1).allclose(torch.ones(n), atol=1e-5), (
        f"Softmax output should sum to 1, but got: {y.sum()}"
    )
