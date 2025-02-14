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
        self.y_true = []
        self.y_pred = []

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

        # Append to the class-global values
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        

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
    
    def __returnmetric__(self):
        if self.y_true == [] and self.y_pred == []:
            return []
        elif self.y_true == [] or self.y_pred == []:
            raise ValueError("y_true or y_pred is empty.")
        self.y_true = torch.cat(self.y_true)
        self.y_pred = torch.cat(self.y_pred)
        
        return self._macro_avg_precision(self.y_true, self.y_pred) if self.macro_averaging else self._micro_avg_precision(self.y_true, self.y_pred)    
    
    def __reset__(self):
        """Resets the class-global lists of true and predicted values to empty lists.

        Returns
        -------
        None
            Returns None
        """
        self.y_true = self.y_pred = []
        return None


if __name__ == "__main__":
    print(
        "Congratulations, you succesfully ran the Precision metric class. You should be proud of this marvelous achievement!"
    )
