import copy

import numpy as np
import torch.nn as nn

from .metrics import EntropyPrediction, F1Score, precision


class MetricWrapper(nn.Module):
    """
    Wrapper class for metrics, that runs multiple metrics on the same data.

    Args
    ----
        metrics : list[str]
            List of metrics to run on the data.

    Attributes
    ----------
        metrics : dict
            Dictionary containing the metric functions.
        tmp_scores : dict
            Dictionary containing the temporary scores of the metrics.

    Methods
    -------
        __call__(y_true, y_pred)
            Call the metric functions on the true and predicted labels.
        accumulate()
            Get the average scores of the metrics.
        reset()
            Reset the temporary scores of the metrics.

    Examples
    --------
    >>> from utils import MetricWrapper
    >>> metrics = MetricWrapper("entropy", "f1", "precision")
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 1, 0]
    >>> metrics(y_true, y_pred)
    >>> metrics.accumulate()
    {'entropy': 0.6931471805599453, 'f1': 0.5, 'precision': 0.5}
    >>> metrics.reset()
    >>> metrics.accumulate()
    {'entropy': [], 'f1': [], 'precision': []}
    """

    def __init__(self, *metrics):
        super().__init__()
        self.metrics = {}

        for metric in metrics:
            self.metrics[metric] = self._get_metric(metric)

        self.tmp_scores = copy.deepcopy(self.metrics)
        for key in self.tmp_scores:
            self.tmp_scores[key] = []

    def _get_metric(self, key):
        """
        Get the metric function based on the key

        Args
        ----
            key (str): metric name

        Returns
        -------
            metric (callable): metric function
        """

        match key.lower():
            case "entropy":
                return EntropyPrediction()
            case "f1":
                raise F1Score()
            case "recall":
                raise NotImplementedError("Recall score not implemented yet")
            case "precision":
                return precision()
            case "accuracy":
                raise NotImplementedError("Accuracy score not implemented yet")
            case _:
                raise ValueError(f"Metric {key} not supported")

    def __call__(self, y_true, y_pred):
        for key in self.metrics:
            self.tmp_scores[key].append(self.metrics[key](y_true, y_pred))

    def accumulate(self):
        return_metrics = {}
        for key in self.metrics:
            return_metrics[key] = np.mean(self.tmp_scores[key])

        return return_metrics

    def reset(self):
        for key in self.tmp_scores:
            self.tmp_scores[key] = []
