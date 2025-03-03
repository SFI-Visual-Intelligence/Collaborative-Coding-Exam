import numpy as np
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

        self.__y_true = []
        self.__y_pred = []

    def forward(self, true, logits):
        pred = logits.argmax(dim=-1)
        y_true = one_hot_encode(true, self.num_classes)
        y_pred = one_hot_encode(pred, self.num_classes)

        self.__y_true.append(y_true)
        self.__y_pred.append(y_pred)

    def compute(self, y_true, y_pred):
        if self.macro_averaging:
            return self.__compute_macro_averaging(y_true, y_pred)

        return self.__compute_micro_averaging(y_true, y_pred)

    def __compute_macro_averaging(self, y_true, y_pred):
        recall = 0
        for i in range(self.num_classes):
            tp = (y_true[:, i] * y_pred[:, i]).sum()
            fn = (y_true[:, i] * (1 - y_pred[:, i])).sum()
            recall += tp / (tp + fn)
        recall /= self.num_classes

        return recall

    def __compute_micro_averaging(self, y_true, y_pred):
        true_positives = (y_true * y_pred).sum()
        false_negatives = (y_true * (1 - y_pred)).sum()

        recall = true_positives / (true_positives + false_negatives)
        return recall

    def __returnmetric__(self):
        if len(self.__y_true) == 0 and len(self.__y_pred) == 0:
            return np.nan

        y_true = torch.cat(self.__y_true, dim=0)
        y_pred = torch.cat(self.__y_pred, dim=0)

        return self.compute(y_true, y_pred)

    def __reset__(self):
        self.__y_true = []
        self.__y_pred = []
