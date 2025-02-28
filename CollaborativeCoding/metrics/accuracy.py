import numpy as np
import torch
from torch import nn


class Accuracy(nn.Module):
    """
    Computes the accuracy of a model's predictions.

    Args
    ----------
    num_classes : int
        The number of classes in the classification task.
    macro_averaging : bool, optional
        If True, computes macro-average accuracy. Otherwise, computes micro-average accuracy. Default is False.


    Methods
    -------
    forward(y_true, y_pred)
        Stores the true and predicted labels. Typically called for each batch during the forward pass of a model.
    _macro_acc()
        Computes the macro-average accuracy.
    _micro_acc()
        Computes the micro-average accuracy.
    __returnmetric__()
        Returns the computed accuracy based on the averaging method for all stored predictions.
    __reset__()
        Resets the stored true and predicted labels.

    Examples
    --------
    >>> y_true = torch.tensor([0, 1, 2, 3, 3])
    >>> y_pred = torch.tensor([0, 1, 2, 3, 0])
    >>> accuracy = Accuracy(num_classes=4)
    >>> accuracy(y_true, y_pred)
    >>> accuracy.__returnmetric__()
    0.8
    >>> accuracy.__reset__()
    >>> accuracy.macro_averaging = True
    >>> accuracy(y_true, y_pred)
    >>> accuracy.__returnmetric__()
    0.875
    """

    def __init__(self, num_classes, macro_averaging=False):
        super().__init__()
        self.num_classes = num_classes
        self.macro_averaging = macro_averaging
        self.y_true = []
        self.y_pred = []

    def forward(self, y_true, y_pred):
        """
        Store the true and predicted labels.

        Parameters
        ----------
        y_true : torch.Tensor
            True labels.
        y_pred : torch.Tensor
            Predicted labels. Either a 1D tensor of shape (batch_size,) or a 2D tensor of shape (batch_size, num_classes).
        """
        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=1)
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def _macro_acc(self):
        """
        Compute the macro-average accuracy on the stored predictions.

        Returns
        -------
        float
            Macro-average accuracy score.
        """
        y_true, y_pred = self.y_true.flatten(), self.y_pred.flatten()  # Ensure 1D shape

        classes = torch.unique(y_true)  # Find unique class labels
        acc_per_class = []

        for c in classes:
            mask = y_true == c  # Mask for class c
            acc = (y_pred[mask] == y_true[mask]).float().mean()  # Accuracy for class c
            acc_per_class.append(acc)

        macro_acc = torch.stack(acc_per_class).mean().item()  # Average across classes
        return macro_acc

    def _micro_acc(self):
        """
        Compute the micro-average accuracy on the stored predictions.

        Returns
        -------
        float
            Micro-average accuracy score.
        """
        return (self.y_true == self.y_pred).float().mean().item()

    def __returnmetric__(self):
        """
        Return the computed accuracy based on the averaging method for all stored predictions.

        Returns
        -------
        float
            Computed accuracy score.
        """
        if self.y_true == [] or self.y_pred == []:
            return np.nan
        if isinstance(self.y_true, list):
            if len(self.y_true) == 1:
                self.y_true = self.y_true[0]
                self.y_pred = self.y_pred[0]
            else:
                self.y_true = torch.cat(self.y_true)
                self.y_pred = torch.cat(self.y_pred)
        return self._micro_acc() if not self.macro_averaging else self._macro_acc()

    def __reset__(self):
        """
        Reset the stored true and predicted labels.
        """
        self.y_true = []
        self.y_pred = []
        return None
