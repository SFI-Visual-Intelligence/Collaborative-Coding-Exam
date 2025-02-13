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
        print(self.y_true, self.y_pred)
        return (self.y_true == self.y_pred).float().mean().item()
    
    def __returnmetric__(self):
        print(self.y_true, self.y_pred)
        print(self.y_true == [], self.y_pred == [])
        print(len(self.y_true), len(self.y_pred))
        print(type(self.y_true), type(self.y_pred))
        if self.y_true == [] or self.y_pred == []:
            return 0.0
        if isinstance(self.y_true,list):
            if len(self.y_true) == 1:
                self.y_true = self.y_true[0]
                self.y_pred = self.y_pred[0]
            else:
                self.y_true = torch.cat(self.y_true)
                self.y_pred = torch.cat(self.y_pred)
        return self._micro_acc() if not self.macro_averaging else self._macro_acc()
    
    def __resetmetric__(self):
        self.y_true = []
        self.y_pred = []
        return None


if __name__ == "__main__":
    # Test the accuracy metric
    y_true = torch.tensor([0, 1, 2, 3, 4, 5])
    y_pred = torch.tensor([0, 1, 2, 3, 4, 5])
    accuracy = Accuracy(num_classes=6, macro_averaging=False)
    accuracy(y_true, y_pred)
    print(accuracy.__returnmetric__())  # 1.0
    accuracy.__resetmetric__()
    print(accuracy.__returnmetric__())  # 0.0
    y_pred = torch.tensor([0, 1, 2, 3, 4, 4])
    accuracy(y_true, y_pred)
    print(accuracy.__returnmetric__())  # 0.8333333134651184
    accuracy.__resetmetric__()
    print(accuracy.__returnmetric__())  # 0.0
    accuracy.macro_averaging = True
    accuracy(y_true, y_pred)
    y_true_1 = torch.tensor([0, 1, 2, 3, 4, 5])
    y_pred_1 = torch.tensor([0, 1, 2, 3, 4, 4])
    accuracy(y_true_1, y_pred_1)
    print(accuracy.__returnmetric__())  # 0.9166666865348816
    #accuracy.__resetmetric__()
    #accuracy(y_true, y_pred)
    #accuracy(y_true_1, y_pred_1)
    accuracy.macro_averaging = False
    print(accuracy.__returnmetric__())  # 0.8333333134651184
    accuracy.__resetmetric__()
    print(accuracy.__returnmetric__())  # 0.0
    print(accuracy.__resetmetric__())  # None
