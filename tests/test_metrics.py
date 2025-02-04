from utils.metrics import F1Score, Recall


def test_recall():
    import torch

    recall = Recall(7)

    y_true = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    y_pred = torch.tensor([2, 1, 2, 1, 4, 5, 6])

    recall_score = recall(y_true, y_pred)

    assert recall_score.allclose(torch.tensor(0.7143), atol=1e-5), (
        f"Recall Score: {recall_score.item()}"
    )


def test_f1score():
    import torch

    f1_metric = F1Score(num_classes=3)
    preds = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]
    )

    target = torch.tensor([0, 1, 0, 2])

    f1_metric.update(preds, target)
    assert f1_metric.tp.sum().item() > 0, "Expected some true positives."
    assert f1_metric.fp.sum().item() > 0, "Expected some false positives."
    assert f1_metric.fn.sum().item() > 0, "Expected some false negatives."
