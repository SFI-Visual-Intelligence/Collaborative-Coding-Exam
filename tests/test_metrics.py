from random import randint

import pytest

from utils.load_metric import MetricWrapper
from utils.metrics import Accuracy, F1Score, Precision, Recall


@pytest.mark.parametrize(
    "metric, num_classes, macro_averaging",
    [
        ("f1", randint(2, 10), False),
        ("f1", randint(2, 10), True),
        ("recall", randint(2, 10), False),
        ("recall", randint(2, 10), True),
        ("accuracy", randint(2, 10), False),
        ("accuracy", randint(2, 10), True),
        ("precision", randint(2, 10), False),
        ("precision", randint(2, 10), True),
        # TODO: Add test for EntropyPrediction
    ],
)
def test_metric_wrapper(metric, num_classes, macro_averaging):
    import numpy as np
    import torch

    y_true = torch.arange(num_classes, dtype=torch.int64)
    logits = torch.rand(num_classes, num_classes)

    metrics = MetricWrapper(
        metric,
        num_classes=num_classes,
        macro_averaging=macro_averaging,
    )

    metrics(y_true, logits)
    score = metrics.accumulate()
    metrics.reset()
    empty_score = metrics.accumulate()

    assert isinstance(score, dict), "Expected a dictionary output."
    assert metric in score, f"Expected {metric} metric in the output."
    assert score[metric] >= 0, "Expected a non-negative value."
    assert np.isnan(empty_score[metric]), "Expected an empty list."


def test_recall():
    import torch

    y_true = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    logits = torch.randn(7, 7)

    recall_micro = Recall(7)
    recall_macro = Recall(7, macro_averaging=True)

    recall_micro_score = recall_micro(y_true, logits)
    recall_macro_score = recall_macro(y_true, logits)

    assert isinstance(recall_micro_score, torch.Tensor), "Expected a tensor output."
    assert isinstance(recall_macro_score, torch.Tensor), "Expected a tensor output."
    assert recall_micro_score.item() >= 0, "Expected a non-negative value."
    assert recall_macro_score.item() >= 0, "Expected a non-negative value."


def test_f1score():
    import torch

    f1_metric = F1Score(num_classes=3)
    preds = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]
    )

    target = torch.tensor([0, 1, 0, 2])

    f1_metric(preds, target)
    assert f1_metric.tp.sum().item() > 0, "Expected some true positives."
    assert f1_metric.fp.sum().item() > 0, "Expected some false positives."
    assert f1_metric.fn.sum().item() > 0, "Expected some false negatives."


def test_precision_case1():
    import torch

    for boolean, true_precision in zip([False, True], [25.0 / 36, 7.0 / 10]):
        true1 = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])
        pred1 = torch.tensor([0, 2, 1, 1, 0, 2, 0, 0, 2, 1])
        P = Precision(3, micro_averaging=boolean)
        precision1 = P(true1, pred1)
        assert precision1.allclose(torch.tensor(true_precision), atol=1e-5), (
            f"Precision Score: {precision1.item()}"
        )


def test_precision_case2():
    import torch

    for boolean, true_precision in zip([False, True], [8.0 / 15, 6.0 / 15]):
        true2 = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        pred2 = torch.tensor([0, 0, 4, 3, 4, 0, 4, 4, 2, 3, 4, 1, 2, 4, 0])
        P = Precision(5, micro_averaging=boolean)
        precision2 = P(true2, pred2)
        assert precision2.allclose(torch.tensor(true_precision), atol=1e-5), (
            f"Precision Score: {precision2.item()}"
        )


def test_precision_case3():
    import torch

    for boolean, true_precision in zip([False, True], [3.0 / 4, 4.0 / 5]):
        true3 = torch.tensor([0, 0, 0, 1, 0])
        pred3 = torch.tensor([1, 0, 0, 1, 0])
        P = Precision(2, micro_averaging=boolean)
        precision3 = P(true3, pred3)
        assert precision3.allclose(torch.tensor(true_precision), atol=1e-5), (
            f"Precision Score: {precision3.item()}"
        )


def test_for_zero_denominator():
    import torch

    for boolean in [False, True]:
        true4 = torch.tensor([1, 1, 1, 1, 1])
        pred4 = torch.tensor([0, 0, 0, 0, 0])
        P = Precision(2, micro_averaging=boolean)
        precision4 = P(true4, pred4)
        assert precision4.allclose(torch.tensor(0.0), atol=1e-5), (
            f"Precision Score: {precision4.item()}"
        )


def test_accuracy():
    import torch

    accuracy = Accuracy(num_classes=5)

    y_true = torch.tensor([0, 3, 2, 3, 4])
    y_pred = torch.tensor([0, 1, 2, 3, 4])

    accuracy_score = accuracy(y_true, y_pred)

    assert torch.abs(torch.tensor(accuracy_score - 0.8)) < 1e-5, (
        f"Accuracy Score: {accuracy_score.item()}"
    )
