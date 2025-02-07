import torch.nn as nn


class EntropyPrediction(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_false_logits):
        return

    def __reset__(self):
        pass
