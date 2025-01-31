import torch
import torch.nn as nn


def one_hot_encode(y_true, num_classes):
    """One-hot encode the target tensor.

    Args
    ----
    y_true : torch.Tensor
        Target tensor.
    num_classes : int
        Number of classes in the dataset.

    Returns
    -------
    torch.Tensor
        One-hot encoded tensor.
    """

    y_onehot = torch.zeros(y_true.size(0), num_classes)
    y_onehot.scatter_(1, y_true.unsqueeze(1), 1)
    return y_onehot


class Recall(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        true_onehot = one_hot_encode(y_true, self.num_classes)
        pred_onehot = one_hot_encode(y_pred, self.num_classes)

        true_positives = (true_onehot * pred_onehot).sum()

        false_negatives = torch.sum(~pred_onehot[true_onehot.bool()].bool())

        recall = true_positives / (true_positives + false_negatives)

        return recall


def test_recall():
    recall = Recall(7)

    y_true = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    y_pred = torch.tensor([2, 1, 2, 1, 4, 5, 6])

    recall_score = recall(y_true, y_pred)

    assert recall_score.allclose(torch.tensor(0.7143), atol=1e-5), f"Recall Score: {recall_score.item()}"


def test_one_hot_encode():
    num_classes = 7

    y_true = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    y_onehot = one_hot_encode(y_true, num_classes)

    assert y_onehot.shape == (7, 7), f"Shape: {y_onehot.shape}"
