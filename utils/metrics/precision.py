import torch
import torch.nn as nn


class Precision(nn.Module):
    """Metric module for precision. Can calculate precision both as a mean of precisions or as brute function of true positives and false positives.

    Parameters
    ----------
    num_classes : int
        Number of classes in the dataset.
    micro_averaging : bool
        Wheter to compute the micro or macro precision (default False)
    """

    def __init__(self, num_classes: int, macro_averaging: bool = False):
        super().__init__()

        self.num_classes = num_classes
        self.macro_averaging = macro_averaging

    def forward(self, y_true: torch.tensor, logits: torch.tensor) -> torch.tensor:
        """Compute precision of model

        Parameters
        ----------
        y_true : torch.tensor
            True labels
        y_pred : torch.tensor
            Predicted labels

        Returns
        -------
        torch.tensor
            Precision score
        """
        y_pred = logits.argmax(dim=-1)
        return (
            self._macro_avg_precision(y_true, y_pred)
            if self.macro_averaging
            else self._micro_avg_precision(y_true, y_pred)
        )

    def _micro_avg_precision(
        self, y_true: torch.tensor, y_pred: torch.tensor
    ) -> torch.tensor:
        """Compute micro-average precision by first calculating true/false positive across all classes and then find the precision.

        Parameters
        ----------
        y_true : torch.tensor
            True labels
        y_pred : torch.tensor
            Predicted labels

        Returns
        -------
        torch.tensor
            Micro-averaged precision
        """
        print(y_true.shape)
        true_oh = torch.zeros(y_true.size(0), self.num_classes).scatter_(
            1, y_true.unsqueeze(1), 1
        )
        pred_oh = torch.zeros(y_pred.size(0), self.num_classes).scatter_(
            1, y_pred.unsqueeze(1), 1
        )
        tp = torch.sum(true_oh * pred_oh)
        fp = torch.sum(~true_oh[pred_oh.bool()].bool())

        return torch.nanmean(tp / (tp + fp))

    def _macro_avg_precision(
        self, y_true: torch.tensor, y_pred: torch.tensor
    ) -> torch.tensor:
        """Compute macro-average precision by finding true/false positives of each class separately then averaging across all classes.

        Parameters
        ----------
        y_true : torch.tensor
            True labels
        y_pred : torch.tensor
            Predicted labels

        Returns
        -------
        torch.tensor
            Macro-averaged precision
        """
        true_oh = torch.zeros(y_true.size(0), self.num_classes).scatter_(
            1, y_true.unsqueeze(1), 1
        )
        pred_oh = torch.zeros(y_pred.size(0), self.num_classes).scatter_(
            1, y_pred.unsqueeze(1), 1
        )
        tp = torch.sum(true_oh * pred_oh, 0)
        fp = torch.sum(~true_oh.bool() * pred_oh, 0)

        return torch.nanmean(tp / (tp + fp))


if __name__ == "__main__":
    print(
        "Congratulations, you succesfully ran the Precision metric class. You should be proud of this marvelous achievement!"
    )
