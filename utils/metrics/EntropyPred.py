import numpy as np
import torch.nn as nn
from scipy.stats import entropy


class EntropyPrediction(nn.Module):
    def __init__(self, averages: str = 'average'):
        """
        Initializes the EntropyPrediction module.
        Args:
            averages (str): Specifies the method of aggregation for entropy values. 
                            Must be either 'average' or 'sum'.
        Raises:
            AssertionError: If the averages parameter is not 'average' or 'sum'.
        """
        super().__init__()
        
        assert averages == 'average' or averages == 'sum'
        self.averages = averages
        self.stored_entropy_values = []
        
    def __call__(self, y_true, y_false_logits):
        """
        Computes the entropy between true labels and predicted logits, storing the results.
        Args:
            y_true: The true labels.
            y_false_logits: The predicted logits.
        Side Effects:
            Appends the computed entropy values to the stored_entropy_values list.
        """
        entropy_values = entropy(y_true, qk=y_false_logits)
        return entropy_values
