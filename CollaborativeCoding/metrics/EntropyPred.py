import numpy as np
import torch as th
import torch.nn as nn
from scipy.stats import entropy


class EntropyPrediction(nn.Module):
    def __init__(self, num_classes, macro_averaging=None):
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

        self.stored_entropy_values = []
        self.num_classes = num_classes

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

        assert len(y_logits.size()) == 2, f"y_logits shape: {y_logits.size()}"
        assert y_logits.size(-1) == self.num_classes, (
            f"y_logit class length: {y_logits.size(-1)}, expected: {self.num_classes}"
        )
        y_pred = nn.Softmax(dim=1)(y_logits)
        entropy_values = entropy(y_pred, axis=1)
        entropy_values = th.from_numpy(entropy_values)

        # Fix numerical errors for perfect guesses
        entropy_values[entropy_values == th.inf] = 0
        entropy_values = th.nan_to_num(entropy_values)
        for sample in entropy_values:
            self.stored_entropy_values.append(sample.item())

    def __returnmetric__(self):
        stored_entropy_values = th.from_numpy(np.asarray(self.stored_entropy_values))
        stored_entropy_values = th.mean(stored_entropy_values)

        return stored_entropy_values

    def __reset__(self):
        self.stored_entropy_values = []
