import torch
import torch.nn as nn


def one_hot_encode(vec, num_classes):
    """One-hot encode the target tensor.

    Args
    ----
    vec : torch.Tensor
        Target tensor.
    num_classes : int
        Number of classes in the dataset.

    Returns
    -------
    torch.Tensor
        One-hot encoded tensor.
    """

    onehot = torch.zeros(vec.size(0), num_classes)
    onehot.scatter_(1, vec.unsqueeze(1), 1)
    return onehot


class Recall(nn.Module):
    """
    Recall metric.

    Args
    ----
    num_classes : int
        Number of classes in the dataset.
    macro_averaging : bool
        If True, calculate the recall for each class and return the average.
        If False, calculate the recall for the entire dataset.

    Methods
    -------
    forward(y_true, y_pred)
        Compute the recall metric.

    Examples
    --------
    >>> y_true = torch.tensor([0, 1, 2, 3, 4])
    >>> y_pred = torch.randn(5, 5).argmax(dim=-1)
    >>> recall = Recall(num_classes=5)
    >>> recall(y_true, y_pred)
    0.2
    >>> recall = Recall(num_classes=5, macro_averaging=True)
    >>> recall(y_true, y_pred)
    0.2
    """

    def __init__(self, num_classes, macro_averaging=False):
        super().__init__()
        self.num_classes = num_classes
        self.macro_averaging = macro_averaging

    def forward(self, true, logits):
        pred = logits.argmax(dim=-1)
        y_true = one_hot_encode(true, self.num_classes)
        y_pred = one_hot_encode(pred, self.num_classes)

        if self.macro_averaging:
            recall = 0
            for i in range(self.num_classes):
                tp = (y_true[:, i] * y_pred[:, i]).sum()
                fn = torch.sum(~y_pred[y_true[:, i].bool()].bool())
                recall += tp / (tp + fn)
            recall /= self.num_classes
        else:
            recall = self.__compute(y_true, y_pred)

        return recall

    def __compute(self, y_true, y_pred):
        true_positives = (y_true * y_pred).sum()
        false_negatives = torch.sum(~y_pred[y_true.bool()].bool())

        recall = true_positives / (true_positives + false_negatives)
        return recall
