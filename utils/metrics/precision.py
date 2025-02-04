import torch
import torch.nn as nn

USE_MEAN = True

# Precision = TP / (TP + FP)


class Precision(nn.Module):
    """Metric module for precision. Can calculate precision both as a mean of precisions or as brute function of true positives and false positives. This is for now controller with the USE_MEAN macro.

    Parameters
    ----------
    num_classes : int
        Number of classes in the dataset.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        """Calculates the precision score given number of classes and the true and predicted labels.

        Parameters
        ----------
        y_true : torch.tensor
            true labels
        y_pred : torch.tensor
            predicted labels

        Returns
        -------
        torch.tensor
            precision score
        """
        # One-hot encode the target tensor
        true_oh = torch.zeros(y_true.size(0), self.num_classes).scatter_(
            1, y_true.unsqueeze(1), 1
        )
        pred_oh = torch.zeros(y_pred.size(0), self.num_classes).scatter_(
            1, y_pred.unsqueeze(1), 1
        )

        if USE_MEAN:
            tp = torch.sum(true_oh * pred_oh, 0)
            fp = torch.sum(~true_oh.bool() * pred_oh, 0)

        else:
            tp = torch.sum(true_oh * pred_oh)
            fp = torch.sum(~true_oh[pred_oh.bool()].bool())

        return torch.nanmean(tp / (tp + fp))


def test_precision_case1():
    true_precision = 25.0 / 36 if USE_MEAN else 7.0 / 10

    true1 = torch.tensor([0, 1, 2, 1, 0, 2, 1, 0, 2, 1])
    pred1 = torch.tensor([0, 2, 1, 1, 0, 2, 0, 0, 2, 1])
    P = Precision(3)
    precision1 = P(true1, pred1)
    assert precision1.allclose(torch.tensor(true_precision), atol=1e-5), (
        f"Precision Score: {precision1.item()}"
    )


def test_precision_case2():
    true_precision = 8.0 / 15 if USE_MEAN else 6.0 / 15

    true2 = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    pred2 = torch.tensor([0, 0, 4, 3, 4, 0, 4, 4, 2, 3, 4, 1, 2, 4, 0])
    P = Precision(5)
    precision2 = P(true2, pred2)
    assert precision2.allclose(torch.tensor(true_precision), atol=1e-5), (
        f"Precision Score: {precision2.item()}"
    )


def test_precision_case3():
    true_precision = 3.0 / 4 if USE_MEAN else 4.0 / 5

    true3 = torch.tensor([0, 0, 0, 1, 0])
    pred3 = torch.tensor([1, 0, 0, 1, 0])
    P = Precision(2)
    precision3 = P(true3, pred3)
    assert precision3.allclose(torch.tensor(true_precision), atol=1e-5), (
        f"Precision Score: {precision3.item()}"
    )


def test_for_zero_denominator():
    true_precision = 0.0
    true4 = torch.tensor([1, 1, 1, 1, 1])
    pred4 = torch.tensor([0, 0, 0, 0, 0])
    P = Precision(2)
    precision4 = P(true4, pred4)
    assert precision4.allclose(torch.tensor(true_precision), atol=1e-5), (
        f"Precision Score: {precision4.item()}"
    )


if __name__ == "__main__":
    pass
