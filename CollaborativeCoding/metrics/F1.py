import numpy as np
import torch
import torch.nn as nn


class F1Score(nn.Module):
    """
    F1 Score implementation with support for both macro and micro averaging.

    This class computes the F1 score during training using either macro or micro averaging.
    The F1 score is calculated based on the true positives (TP), false positives (FP),
    and false negatives (FN) for each class.

    Parameters
    ----------
    num_classes : int
        The number of classes in the classification task.

    macro_averaging : bool, optional, default=False
        If True, computes the macro-averaged F1 score. If False, computes the micro-averaged F1 score.

    Attributes
    ----------
    num_classes : int
        The number of classes in the classification task.

    tp : torch.Tensor
        Tensor storing the count of True Positives (TP) for each class.

    fp : torch.Tensor
        Tensor storing the count of False Positives (FP) for each class.

    fn : torch.Tensor
        Tensor storing the count of False Negatives (FN) for each class.

    macro_averaging : bool
        A flag indicating whether to compute the macro-averaged F1 score or not.
    """

    def __init__(self, num_classes, macro_averaging=False):
        """
        Initializes the F1Score object, setting up the necessary state variables.

        Parameters
        ----------
        num_classes : int
            The number of classes in the classification task.

        macro_averaging : bool, optional, default=False
            If True, computes the macro-averaged F1 score. If False, computes the micro-averaged F1 score.
        """
        super().__init__()

        self.num_classes = num_classes
        self.macro_averaging = macro_averaging
        self.y_true = []
        self.y_pred = []
        # Initialize variables for True Positives (TP), False Positives (FP), and False Negatives (FN)
        self.tp = torch.zeros(num_classes)
        self.fp = torch.zeros(num_classes)
        self.fn = torch.zeros(num_classes)

    def _micro_F1(self, target, preds):
        """
        Compute the Micro F1 score by aggregating TP, FP, and FN across all classes.

        Micro F1 score is calculated globally by considering all predictions together, regardless of class.

        Returns
        -------
        torch.Tensor
            The micro-averaged F1 score.
        """
        for i in range(self.num_classes):
            self.tp[i] += torch.sum((preds == i) & (target == i)).float()
            self.fp[i] += torch.sum((preds == i) & (target != i)).float()
            self.fn[i] += torch.sum((preds != i) & (target == i)).float()

        tp = torch.sum(self.tp)
        fp = torch.sum(self.fp)
        fn = torch.sum(self.fn)

        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-8)  # Avoid division by zero

        f1 = (
            2 * precision * recall / (precision + recall + 1e-8)
        )  # Avoid division by zero
        return f1

    def _macro_F1(self, target, preds):
        """
        Compute the Macro F1 score by calculating the F1 score per class and averaging.

        Macro F1 score is calculated as the average of per-class F1 scores. This approach treats all classes equally,
        regardless of their frequency.

        Returns
        -------
        torch.Tensor
            The macro-averaged F1 score.
        """
        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) per class
        for i in range(self.num_classes):
            self.tp[i] += torch.sum((preds == i) & (target == i)).float()
            self.fp[i] += torch.sum((preds == i) & (target != i)).float()
            self.fn[i] += torch.sum((preds != i) & (target == i)).float()

        precision_per_class = self.tp / (
            self.tp + self.fp + 1e-8
        )  # Avoid division by zero
        recall_per_class = self.tp / (
            self.tp + self.fn + 1e-8
        )  # Avoid division by zero
        f1_per_class = (
            2
            * precision_per_class
            * recall_per_class
            / (precision_per_class + recall_per_class + 1e-8)
        )  # Avoid division by zero

        # Take the average of F1 scores across all classes
        f1_score = torch.mean(f1_per_class)
        return f1_score

    def forward(self, preds, target):
        """

        Update the True Positives, False Positives, and False Negatives, and compute the F1 score.

        This method computes the F1 score based on the predictions and true labels. It can compute either the
        macro-averaged or micro-averaged F1 score, depending on the `macro_averaging` flag.

        Parameters
        ----------
        preds : torch.Tensor
            Predicted logits or class indices (shape: [batch_size, num_classes]).
            These logits are typically the output of a softmax or sigmoid activation.

        target : torch.Tensor
            True labels (shape: [batch_size]), where each element is an integer representing the true class.

        Returns
        -------
        torch.Tensor
            The computed F1 score (either micro or macro, based on `macro_averaging`).
        """
        preds = torch.argmax(preds, dim=-1)
        self.y_true.append(target)
        self.y_pred.append(preds)

    def __returnmetric__(self):
        if self.y_true == [] or self.y_pred == []:
            return np.nan
        if isinstance(self.y_true, list):
            if len(self.y_true) == 1:
                self.y_true = self.y_true[0]
                self.y_pred = self.y_pred[0]
            else:
                print(self.y_pred[0], self.y_pred[1])
                self.y_true = torch.cat(self.y_true)
                self.y_pred = torch.cat(self.y_pred)
        return (
            self._micro_F1(self.y_true, self.y_pred)
            if not self.macro_averaging
            else self._macro_F1(self.y_true, self.y_pred)
        )

    def __reset__(self):
        self.y_true = []
        self.y_pred = []
        return None
