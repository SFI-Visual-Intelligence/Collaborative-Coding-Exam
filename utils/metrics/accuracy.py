import torch
from torch import nn


class Accuracy(nn.Module):
    def __init__(self, num_classes, macro_averaging=False):
        super().__init__()
        self.num_classes = num_classes
        self.macro_averaging = macro_averaging

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
        if self.macro_averaging:
            return self._macro_acc(y_true, y_pred)
        else:
            return self._micro_acc(y_true, y_pred)

    def _macro_acc(self, y_true, y_pred):
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
        y_true, y_pred = y_true.flatten(), y_pred.flatten()  # Ensure 1D shape

        classes = torch.unique(y_true)  # Find unique class labels
        acc_per_class = []

        for c in classes:
            mask = y_true == c  # Mask for class c
            acc = (y_pred[mask] == y_true[mask]).float().mean()  # Accuracy for class c
            acc_per_class.append(acc)

        macro_acc = torch.stack(acc_per_class).mean().item()  # Average across classes
        return macro_acc

    def _micro_acc(self, y_true, y_pred):
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
        return (y_true == y_pred).float().mean().item()


if __name__ == "__main__":
    accuracy = Accuracy(5)
    macro_accuracy = Accuracy(5, macro_averaging=True)

    y_true = torch.tensor([0, 3, 2, 3, 4])
    y_pred = torch.tensor([0, 1, 2, 3, 4])
    print(accuracy(y_true, y_pred))
    print(macro_accuracy(y_true, y_pred))

    y_true = torch.tensor([0, 3, 2, 3, 4])
    y_onehot_pred = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    print(accuracy(y_true, y_onehot_pred))
    print(macro_accuracy(y_true, y_onehot_pred))
