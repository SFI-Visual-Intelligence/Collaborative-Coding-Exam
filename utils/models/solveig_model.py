import pytest
import torch
import torch.nn as nn


class SolveigModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return