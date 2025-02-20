import torch.nn as nn

from .metrics import Accuracy, EntropyPrediction, F1Score, Precision, Recall


class MetricWrapper(nn.Module):
    """
    A wrapper class for evaluating multiple metrics on the same dataset.
    This class allows you to compute several metrics simultaneously on given
    true and predicted labels. It supports a variety of common metrics and
    provides methods to accumulate results and reset the state.
    Args
    ----
        num_classes : int
            The number of classes in the classification task.
        metrics : list[str]
            A list of metric names to be evaluated.
    Attributes
    ----------
        metrics : dict
            A dictionary mapping metric names to their corresponding functions.
        num_classes : int
            The number of classes for the classification task.
    Methods
    -------
        __call__(y_true, y_pred)
            Computes the specified metrics on the provided true and predicted labels.
        getmetrics(str_prefix: str = None)
            Retrieves the computed metrics, optionally prefixed with a string.
        resetmetric()
            Resets the state of all metric computations.
    Examples
    --------
    >>> from CollaborativeCoding import MetricWrapperProposed
    >>> metrics = MetricWrapperProposed(2, "entropy", "f1", "precision")
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 1, 0]
    >>> metrics(y_true, y_pred)
    >>> metrics.getmetrics()
    {'entropy': 0.6931471805599453, 'f1': 0.5, 'precision': 0.5}
    >>> metrics.resetmetric()
    >>> metrics.getmetrics()
    {'entropy': [], 'f1': [], 'precision': []}
    """

    def __init__(self, *metrics, num_classes, macro_averaging=False, **kwargs):
        super().__init__()
        self.metrics = {}
        self.params = {
            "num_classes": num_classes,
            "macro_averaging": macro_averaging,
        } | kwargs

        for metric in metrics:
            self.metrics[metric] = self._get_metric(metric)

    def _get_metric(self, key):
        """
        Retrieves the metric function based on the provided key.
        Args
        ----
            key (str): The name of the metric.
        Returns
        -------
            metric (callable): The function that computes the metric.
        """
        match key.lower():
            case "entropy":
                return EntropyPrediction(**self.params)
            case "f1":
                return F1Score(**self.params)
            case "recall":
                return Recall(**self.params)
            case "precision":
                return Precision(**self.params)
            case "accuracy":
                return Accuracy(**self.params)
            case _:
                raise ValueError(f"Metric {key} not supported")

    def __call__(self, y_true, y_pred):
        for key in self.metrics:
            self.metrics[key](y_true, y_pred)

    def getmetrics(self, str_prefix: str = None):
        return_metrics = {}
        for key in self.metrics:
            if str_prefix is not None:
                return_metrics[str_prefix + key] = self.metrics[key].__returnmetric__()
            else:
                return_metrics[key] = self.metrics[key].__returnmetric__()
        return return_metrics

    def resetmetric(self):
        for key in self.metrics:
            self.metrics[key].__reset__()


if __name__ == "__main__":
    import torch as th

    metrics = ["entropy", "f1", "recall", "precision", "accuracy"]

    class_sizes = [3, 6, 10]
    for class_size in class_sizes:
        y_true = th.rand((5, class_size)).argmax(dim=1)
        y_pred = th.rand((5, class_size))

        metricwrapper = MetricWrapper(
            metric,
            num_classes=class_size,
            macro_averaging=True if class_size % 2 == 0 else False,
        )

        metricwrapper(y_true, y_pred)
        metric = metricwrapper.getmetrics()
        assert metric is not None

        metricwrapper.resetmetric()
        metric2 = metricwrapper.getmetrics()
        assert metric != metric2
