import itertools

import pytest

from CollaborativeCoding.load_metric import MetricWrapper

METRICS = ["f1", "recall", "accuracy", "precision", "entropy"]


def _metric_combinations():
    """
    Yield various combinations of metrics:
      1. Single metric as a list
      2. Pairs of metrics
      3. All metrics
    """

    # Single metrics as lists
    for m in METRICS:
        yield [m]

    # Pairs of metrics (2-combinations)
    for combo in itertools.combinations(METRICS, 2):
        yield list(combo)

    # Also test all metrics at once
    yield METRICS


@pytest.mark.parametrize("metrics", _metric_combinations())
@pytest.mark.parametrize("num_classes", [2, 3, 5, 10])
@pytest.mark.parametrize("macro_averaging", [True, False])
def test_metric_wrapper(metrics, num_classes, macro_averaging):
    import numpy as np
    import torch

    y_true = torch.arange(num_classes, dtype=torch.int64)
    logits = torch.rand(num_classes, num_classes)

    mw = MetricWrapper(
        *metrics,
        num_classes=num_classes,
        macro_averaging=macro_averaging,
    )

    mw(y_true, logits)
    score = mw.getmetrics()
    mw.resetmetric()
    empty_score = mw.getmetrics()

    assert isinstance(score, dict), "Expected a dictionary output."
    for m in metrics:
        assert m in score, f"Expected metric '{m}' in the output."
        assert score[m] >= 0, "Expected a non-negative value."

        assert m in empty_score, f"Expected metric '{m}' in the output."
        assert np.isnan(empty_score[m]), "Expected an empty list."
