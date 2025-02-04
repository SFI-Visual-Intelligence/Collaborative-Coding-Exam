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
