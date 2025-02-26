import numpy as np
import torch
import torch.nn as nn


class F1Score(nn.Module):
    """
    F1 Score implementation with support for both macro and micro averaging.
    This class computes the F1 score during training using either macro or micro averaging.
    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.

    macro_averaging : bool, default=False
        If True, computes the macro-averaged F1 score. If False, computes the micro-averaged F1 score.
    """

    def __init__(self, num_classes, macro_averaging=False):
        super().__init__()
        self.num_classes = num_classes
        self.macro_averaging = macro_averaging
        self.y_true = []
        self.y_pred = []


    def forward(self, target, preds):
        """
        Stores predictions and targets for computing the F1 score.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted logits (shape: [batch_size, num_classes]).
        target : torch.Tensor
            True labels (shape: [batch_size]).
        """
        preds = torch.argmax(preds, dim=-1)  # Convert logits to class indices
        self.y_true.append(target.detach())
        if preds.dim() == 0:  # Scalar (e.g., single class prediction)
            preds = preds.unsqueeze(0)  # Add batch dimension
        self.y_pred.append(preds.detach())

    def compute_f1(self):
        """
        Computes the F1 score (Micro or Macro).

        Returns
        -------
        torch.Tensor
            The computed F1 score.
        """
        if not self.y_true or not self.y_pred:  # Check if empty
            return torch.tensor(np.nan)

        # Convert lists to tensors
        y_true = torch.cat(self.y_true)
        y_pred = torch.cat(self.y_pred)

        return self._macro_F1(y_true, y_pred) if self.macro_averaging else self._micro_F1(y_true, y_pred)

    def _micro_F1(self, target, preds):
        """Computes Micro F1 Score (global TP, FP, FN)."""
        tp = torch.sum(preds == target).float()
        fp = torch.sum(preds != target).float()
        fn = fp  # Since all errors are either FP or FN

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return f1

    def _macro_F1(self, target, preds):
        """Computes Macro F1 Score in a vectorized way (no loops)."""
        num_classes = self.num_classes
        target = target.long()  # Ensure target is a LongTensor
        preds = preds.long()
        # Create one-hot encodings of the true and predicted labels
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes)
        preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=num_classes)

        # Compute TP, FP, FN for each class
        tp = torch.sum(target_one_hot * preds_one_hot, dim=0).float()
        fp = torch.sum(preds_one_hot * (1 - target_one_hot), dim=0).float()
        fn = torch.sum(target_one_hot * (1 - preds_one_hot), dim=0).float()

        # Compute precision and recall per class
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        # Compute per-class F1 score
        f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Compute Macro F1 (mean over all classes)
        return torch.mean(f1_per_class)

    def __returnmetric__(self):
        """
        Computes and returns the F1 score (Micro or Macro).

        Returns
        -------
        torch.Tensor
            The computed F1 score.
        """
        if not self.y_true or not self.y_pred:  # Check if empty
            return torch.tensor(np.nan)

        # Convert lists to tensors
        y_true = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in self.y_true])
        y_pred = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in self.y_pred])

        return self._macro_F1(y_true, y_pred) if self.macro_averaging else self._micro_F1(y_true, y_pred)

    def __reset__(self):
        """Resets stored predictions and targets."""
        self.y_true = []
        self.y_pred = []