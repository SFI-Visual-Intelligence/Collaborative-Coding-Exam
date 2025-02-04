import torch
import torch.nn as nn


class F1Score(nn.Module):
    """
    F1 Score implementation with direct averaging inside the compute method.

    Parameters
    ----------
    num_classes : int
        Number of classes.

    Attributes
    ----------
    num_classes : int
        The number of classes.

    tp : torch.Tensor
        Tensor for True Positives (TP) for each class.

    fp : torch.Tensor
        Tensor for False Positives (FP) for each class.

    fn : torch.Tensor
        Tensor for False Negatives (FN) for each class.
    """

    def __init__(self, num_classes):
        """
        Initializes the F1Score object, setting up the necessary state variables.

        Parameters
        ----------
        num_classes : int
            The number of classes in the classification task.

        """

        super().__init__()

        self.num_classes = num_classes

        # Initialize  variables for True Positives (TP), False Positives (FP), and False Negatives (FN)
        self.tp = torch.zeros(num_classes)
        self.fp = torch.zeros(num_classes)
        self.fn = torch.zeros(num_classes)

    def update(self, preds, target):
        """
        Update the variables with predictions and true labels.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted logits (shape: [batch_size, num_classes]).

        target : torch.Tensor
            True labels (shape: [batch_size]).
        """
        preds = torch.argmax(preds, dim=1)

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) per class
        for i in range(self.num_classes):
            self.tp[i] += torch.sum((preds == i) & (target == i)).float()
            self.fp[i] += torch.sum((preds == i) & (target != i)).float()
            self.fn[i] += torch.sum((preds != i) & (target == i)).float()

    def compute(self):
        """
        Compute the F1 score.

        Returns
        -------
        torch.Tensor
           The computed F1 score.
        """

        # Compute F1 score based on the specified averaging method
        f1_score = (
            2
            * torch.sum(self.tp)
            / (2 * torch.sum(self.tp) + torch.sum(self.fp) + torch.sum(self.fn))
        )

        return f1_score


def test_f1score():
    f1_metric = F1Score(num_classes=3)
    preds = torch.tensor(
        [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]
    )

    target = torch.tensor([0, 1, 0, 2])

    f1_metric.update(preds, target)
    assert f1_metric.tp.sum().item() > 0, "Expected some true positives."
    assert f1_metric.fp.sum().item() > 0, "Expected some false positives."
    assert f1_metric.fn.sum().item() > 0, "Expected some false negatives."
