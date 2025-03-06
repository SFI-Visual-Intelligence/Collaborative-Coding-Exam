import numpy as np
import torch
import torch.nn as nn


class F1Score(nn.Module):
    """
    Computes the F1 score for classification tasks with support for both macro and micro averaging.

    This class allows you to compute the F1 score during training or evaluation. You can select between two methods of averaging:
    - **Micro Averaging**: Computes the F1 score globally, treating each individual prediction as equally important.
    - **Macro Averaging**: Computes the F1 score for each class individually and then averages the scores.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.

    macro_averaging : bool, optional, default=False
        If True, computes the macro-averaged F1 score. If False, computes the micro-averaged F1 score. Default is micro averaging.

    Attributes
    ----------
    num_classes : int
        The number of classes in the classification task.

    macro_averaging : bool
        A flag to determine whether to compute the macro-averaged or micro-averaged F1 score.

    y_true : list
        A list to store true labels for the current batch.

    y_pred : list
        A list to store predicted labels for the current batch.

    Methods
    -------
    forward(target, preds)
        Stores predictions and true labels for computing the F1 score during training or evaluation.

    compute_f1()
        Computes and returns the F1 score based on the stored predictions and true labels.

    _micro_F1(target, preds)
        Computes the micro-averaged F1 score based on the global true positive, false positive, and false negative counts.

    _macro_F1(target, preds)
        Computes the macro-averaged F1 score by calculating the F1 score per class and then averaging across all classes.

    __returnmetric__()
        Computes and returns the F1 score (Micro or Macro) as specified.

    __reset__()
        Resets the stored predictions and true labels, preparing for the next batch or epoch.
    """

    def __init__(self, num_classes, macro_averaging=False):
        """
        Initializes the F1Score object with the number of classes and averaging mode.

        Parameters
        ----------
        num_classes : int
           The number of classes in the classification task.

        macro_averaging : bool, optional, default=False
           If True, compute the macro-averaged F1 score. If False, compute the micro-averaged F1 score.
        """
        super().__init__()
        self.num_classes = num_classes
        self.macro_averaging = macro_averaging
        self.y_true = []
        self.y_pred = []

    def forward(self, target, preds):
        """
        Stores the true labels and predictions to compute the F1 score.

        Parameters
        ----------
        target : torch.Tensor
            True labels (shape: [batch_size]).

        preds : torch.Tensor
            Predicted logits (shape: [batch_size, num_classes]).
        """
        preds = torch.argmax(preds, dim=-1)  # Convert logits to class indices
        self.y_true.append(target.detach())
        if preds.dim() == 0:  # Scalar (e.g., single class prediction)
            preds = preds.unsqueeze(0)  # Add batch dimension
        self.y_pred.append(preds.detach())

    def _micro_F1(self, target, preds):
        """Computes the Micro-averaged F1 score (global TP, FP, FN)."""
        tp = torch.sum(preds == target).float()
        fp = torch.sum(preds != target).float()
        fn = fp  # Since all errors are either FP or FN

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return f1

    def _macro_F1(self, target, preds):
        """Computes the Macro-averaged F1 score."""
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
        Computes and returns the F1 score (Micro or Macro) based on the stored predictions and targets.

        Returns
        -------
        torch.Tensor
            The computed F1 score. Returns NaN if no predictions or targets are available.
        """
        if not self.y_true or not self.y_pred:  # Check if empty
            return torch.tensor(np.nan)

        # Convert lists to tensors
        y_true = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in self.y_true])
        y_pred = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in self.y_pred])

        return (
            self._macro_F1(y_true, y_pred)
            if self.macro_averaging
            else self._micro_F1(y_true, y_pred)
        )

    def __reset__(self):
        """Resets the stored predictions and targets for the next batch or epoch."""
        self.y_true = []
        self.y_pred = []
