import copy

import numpy as np
import torch.nn as nn

from .metrics import Accuracy, EntropyPrediction, F1Score, Precision, Recall


class MetricWrapper(nn.Module):
    def __init__(self, *metrics, num_classes):
        super().__init__()
        self.metrics = {}
        self.num_classes = num_classes

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
                return EntropyPrediction(num_classes=self.num_classes)
            case "f1":
                return F1Score(num_classes=self.num_classes)
            case "recall":
                return Recall(num_classes=self.num_classes)
            case "precision":
                return Precision(num_classes=self.num_classes)
            case "accuracy":
                return Accuracy(num_classes=self.num_classes)
            case _:
                raise ValueError(f"Metric {key} not supported")

    def __call__(self, y_true, y_pred):
        for key in self.metrics:
            self.tmp_scores[key].append(self.metrics[key](y_true, y_pred))

    def __getmetrics__(self, str_prefix: str = None):
        return_metrics = {}
        for key in self.metrics:
            if str_prefix is not None:
                return_metrics[str_prefix + key] = np.mean(self.tmp_scores[key])
            else:
                return_metrics[key] = np.mean(self.tmp_scores[key])

        return return_metrics

    def __resetvalues__(self):
        for key in self.tmp_scores:
            self.tmp_scores[key] = []
