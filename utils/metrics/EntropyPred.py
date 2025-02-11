import torch as th 
import torch.nn as nn
import numpy as np
from scipy.stats import entropy


class EntropyPrediction(nn.Module):
    def __init__(self, averages: str = "mean"):
        """
        Initializes the EntropyPrediction module, which calculates the Shannon Entropy
        of predicted logits and aggregates the results based on the specified method.
        Args:
            averages (str): Specifies the method of aggregation for entropy values.
                            Must be one of 'mean', 'sum', or 'none'.
        Raises:
            AssertionError: If the averages parameter is not 'mean', 'sum', or 'none'.
        """
        super().__init__()

        assert averages in ["mean", "sum", "none"], (
            "averages must be 'mean', 'sum', or 'none'"
        )
        self.averages = averages
        self.stored_entropy_values = []

    def __call__(self, y_true: th.Tensor, y_logits: th.Tensor):
        """
        Computes the Shannon Entropy of the predicted logits and stores the results.
        Args:
            y_true: The true labels. This parameter is not used in the computation
                    but is included for compatibility with certain interfaces.
            y_logits: The predicted logits from which entropy is calculated.
        Returns:
            torch.Tensor: The aggregated entropy value(s) based on the specified
                          method ('mean', 'sum', or 'none').
        """
        
        assert len(y_logits.size()) == 2, f'y_logits shape: {y_logits.size()}'
        y_pred = nn.Softmax(dim=1)(y_logits)
        print(f'y_pred: {y_pred}')
        entropy_values = entropy(y_pred, axis=1)
        entropy_values = th.from_numpy(entropy_values)

        # Fix numerical errors for perfect guesses
        entropy_values[entropy_values == th.inf] = 0
        entropy_values = th.nan_to_num(entropy_values)
        print(f'Entropy Values: {entropy_values}')
        for sample in entropy_values:
            self.stored_entropy_values.append(sample.item())


    def __returnmetric__(self):
        stored_entropy_values = th.from_numpy(np.asarray(self.stored_entropy_values))

        if self.averages == "mean":
            stored_entropy_values = th.mean(stored_entropy_values)
        elif self.averages == "sum":
            stored_entropy_values = th.sum(stored_entropy_values)
        elif self.averages == "none":
            pass 
        return stored_entropy_values

    def __reset__(self):
        self.stored_entropy_values = []

