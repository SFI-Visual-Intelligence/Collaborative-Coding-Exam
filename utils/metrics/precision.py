import torch
import torch.nn as nn


class Precision(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        pass