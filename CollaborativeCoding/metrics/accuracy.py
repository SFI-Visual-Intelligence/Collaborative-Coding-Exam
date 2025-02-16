import numpy as np
import torch
from torch import nn


class Accuracy(nn.Module):
    def __init__(self, num_classes, macro_averaging=False):
        super().__init__()
        self.num_classes = num_classes
        self.macro_averaging = macro_averaging
        self.y_true = []
        self.y_pred = []

    def forward(self, y_true, y_pred):
        """
        Compute the accuracy of the model.

        Parameters
        ----------
        y_true : torch.Tensor
            True labels.
        y_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            Accuracy score.
        """
        if y_pred.dim() > 1:
            y_pred = y_pred.argmax(dim=1)
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)

    def _macro_acc(self):
        """
        Compute the macro-average accuracy.

        Parameters
        ----------
        y_true : torch.Tensor
            True labels.
        y_pred : torch.Tensor
            Predicted labels.

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
        Compute the micro-average accuracy.

        Parameters
        ----------
        y_true : torch.Tensor
            True labels.
        y_pred : torch.Tensor
            Predicted labels.

        Returns
        -------
        float
            Micro-average accuracy score.
        """
        return (self.y_true == self.y_pred).float().mean().item()

    def __returnmetric__(self):
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
        self.y_true = []
        self.y_pred = []
        return None
