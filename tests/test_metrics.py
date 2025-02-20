from random import randint

import pytest

from CollaborativeCoding.load_metric import MetricWrapper
from CollaborativeCoding.metrics import (
    Accuracy,
    EntropyPrediction,
    F1Score,
    Precision,
    Recall,
)


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
        ("entropy", randint(2, 10), False),
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
    score = metrics.getmetrics()
    metrics.resetmetric()
    empty_score = metrics.getmetrics()

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

    recall_micro(y_true, logits)
    recall_macro(y_true, logits)

    recall_micro_score = recall_micro.__returnmetric__()
    recall_macro_score = recall_macro.__returnmetric__()

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


def test_precision():
    import numpy as np
    import torch
    from sklearn.metrics import precision_score

    C = randint(2, 10)  # number of classes
    N = randint(2, 10 * C)  # batchsize
    y_true = torch.randint(0, C, (N,))
    logits = torch.randn(N, C)

    # create metric objects
    precision_micro = Precision(num_classes=C)
    precision_macro = Precision(num_classes=C, macro_averaging=True)

    # run metric object
    precision_micro(y_true, logits)
    precision_macro(y_true, logits)

    # get metric scores
    micro_precision_score = precision_micro.__returnmetric__()
    macro_precision_score = precision_macro.__returnmetric__()

    # check output to be tensor
    assert isinstance(micro_precision_score, torch.Tensor), "Tensor output is expected."
    assert isinstance(macro_precision_score, torch.Tensor), "Tensor output is expected."

    # check for non-negativity
    assert micro_precision_score.item() >= 0, "Expected non-negative value"
    assert macro_precision_score.item() >= 0, "Expected non-negative value"

    # find predictions
    y_pred = logits.argmax(dim=-1)

    # check dimension
    assert y_true.shape == torch.Size([N])
    assert logits.shape == torch.Size([N, C])
    assert y_pred.shape == torch.Size([N])

    # find true values with scikit learn
    scikit_macro_precision = precision_score(y_true, y_pred, average="macro")
    scikit_micro_precision = precision_score(y_true, y_pred, average="micro")

    # check for similarity
    assert np.isclose(scikit_micro_precision, micro_precision_score, atol=1e-5), (
        "Score does not match scikit's score"
    )
    assert np.isclose(scikit_macro_precision, macro_precision_score, atol=1e-5), (
        "Score does not match scikit's score"
    )


def test_accuracy():
    import numpy as np
    import torch

    # Test the accuracy metric
    y_true = torch.tensor([0, 1, 2, 3, 4, 5])
    y_pred = torch.tensor([0, 1, 2, 3, 4, 5])
    accuracy = Accuracy(num_classes=6, macro_averaging=False)
    accuracy(y_true, y_pred)
    assert accuracy.__returnmetric__() == 1.0, "Expected accuracy to be 1.0"
    accuracy.__reset__()
    assert accuracy.__returnmetric__() is np.nan, "Expected accuracy to be 0.0"
    y_pred = torch.tensor([0, 1, 2, 3, 4, 4])
    accuracy(y_true, y_pred)
    assert np.abs(accuracy.__returnmetric__() - 0.8333333134651184) < 1e-5, (
        "Expected accuracy to be 0.8333333134651184"
    )
    accuracy.__reset__()
    accuracy.macro_averaging = True
    accuracy(y_true, y_pred)
    y_true_1 = torch.tensor([0, 1, 2, 3, 4, 5])
    y_pred_1 = torch.tensor([0, 1, 2, 3, 4, 4])
    accuracy(y_true_1, y_pred_1)
    assert np.abs(accuracy.__returnmetric__() - 0.8333333134651184) < 1e-5, (
        "Expected accuracy to be 0.8333333134651186"
    )
    accuracy.macro_averaging = False
    assert np.abs(accuracy.__returnmetric__() - 0.8333333134651184) < 1e-5, (
        "Expected accuracy to be 0.8333333134651184"
    )
    accuracy.__reset__()


def test_entropypred():
    import torch as th

    true_lab = th.rand(6, 5)

    metric = EntropyPrediction(num_classes=5)

    # Test if the metric stores multiple values
    pred_logits = th.rand(6, 5)
    metric(true_lab, pred_logits)

    pred_logits = th.rand(6, 5)
    metric(true_lab, pred_logits)

    pred_logits = th.rand(6, 5)
    metric(true_lab, pred_logits)

    assert type(metric.__returnmetric__()) == th.Tensor

    # Test than an error is raised with num_class != class dimension length
    with pytest.raises(AssertionError):
        metric(true_lab, th.rand(6, 6))
