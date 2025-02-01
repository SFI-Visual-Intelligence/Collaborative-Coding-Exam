from utils.metrics import Recall


def test_recall():
    import torch

    recall = Recall(7)

    y_true = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    y_pred = torch.tensor([2, 1, 2, 1, 4, 5, 6])

    recall_score = recall(y_true, y_pred)

    assert recall_score.allclose(torch.tensor(0.7143), atol=1e-5), (
        f"Recall Score: {recall_score.item()}"
    )
