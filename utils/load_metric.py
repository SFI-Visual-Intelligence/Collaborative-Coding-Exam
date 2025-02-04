import copy

import numpy as np
import torch.nn as nn

from .metrics import EntropyPrediction, F1Score, precision



class MetricWrapper(nn.Module):
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

    def __getmetrics__(self):
        return_metrics = {}
        for key in self.metrics:
            return_metrics[key] = np.mean(self.tmp_scores[key])

        return return_metrics

    def __resetvalues__(self):
        for key in self.tmp_scores:
            self.tmp_scores[key] = []
