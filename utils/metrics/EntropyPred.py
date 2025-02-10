import torch.nn as nn
from scipy.stats import entropy


class EntropyPrediction(nn.Module):
    def __init__(self, averages: str = "average"):
        """
        Initializes the EntropyPrediction module.
        Args:
            averages (str): Specifies the method of aggregation for entropy values.
                            Must be either 'mean', 'sum' or 'none.
        Raises:
            AssertionError: If the averages parameter is not 'mean' or 'sum'.
        """
        super().__init__()

        assert averages == "mean" or averages == "sum"
        self.averages = averages
        self.stored_entropy_values = []

    def __call__(self, y_true, y_logits):
        """
        Computes the Shannon Entropy of the predicted logits, storing the results.
        Args:
            y_true: The true labels. Does nothing, but needed for compatability sake.
            y_logits: The predicted logits.
        """
        entropy_values = entropy(y_logits, axis=1)
        entropy_values = th.from_numpy(entropy_values)

        # Fix numerical errors for perfect guesses
        entropy_values[entropy_values == th.inf] = 0
        entropy_values = th.nan_to_num(entropy_values)

        if self.averages == "mean":
            entropy_values = th.mean(entropy_values)

        elif self.averages == "sum":
            entropy_values = th.sum(entropy_values)

        elif self.averages == "none":
            return entropy_values

        return entropy_values


if __name__ == "__main__":
    import torch as th

    metric = EntropyPrediction(averages="mean")

    true_lab = th.Tensor([0, 1, 1, 2, 4, 3]).reshape(6, 1)
    pred_logits = th.nn.functional.one_hot(true_lab, 5)

    assert th.abs((th.sum(metric(true_lab, pred_logits)) - 0.0)) < 1e-5

    pred_logits = th.rand(6, 5)
    metric2 = EntropyPrediction(averages="sum")
    assert (
        th.abs(
            th.sum(6 * metric(true_lab, pred_logits) - metric2(true_lab, pred_logits))
        )
        < 1e-5
    )
