import torch
import torch.nn as nn

USE_MEAN = True

# Precision = TP / (TP + FP)


class Precision(nn.Module):
    """Metric module for precision. Can calculate precision both as a mean of precisions or as brute function of true positives and false positives.

    Parameters
    ----------
    num_classes : int
        Number of classes in the dataset.
    use_mean : bool
        Whether to calculate precision as a mean of precisions or as a brute function of true positives and false positives.
    """

    def __init__(self, num_classes: int, use_mean: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.use_mean = use_mean

    def forward(self, y_true: torch.tensor, y_pred: torch.tensor) -> torch.tensor:
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

        if self.use_mean:
            tp = torch.sum(true_oh * pred_oh, 0)
            fp = torch.sum(~true_oh.bool() * pred_oh, 0)

        else:
            tp = torch.sum(true_oh * pred_oh)
            fp = torch.sum(~true_oh[pred_oh.bool()].bool())

        return torch.nanmean(tp / (tp + fp))


if __name__ == "__main__":
    pass
