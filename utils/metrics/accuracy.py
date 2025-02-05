import torch
from torch import nn


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

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
        return (y_true == y_pred).float().mean().item()
    
if __name__ == "__main__":
    y_true = torch.tensor([0, 3, 2, 3, 4])
    y_pred = torch.tensor([0, 1, 2, 3, 4])

    accuracy = Accuracy()
    print(accuracy(y_true, y_pred))