import torch.nn as nn
from models import MagnusModel


def load_model(modelname: str) -> nn.Module:
    if modelname == "MagnusModel":
        return MagnusModel()
    else:
        raise ValueError(
            f"Model: {modelname} has not been implemented. \nCheck the documentation for implemented metrics, or check your spelling"
        )
