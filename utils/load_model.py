import torch.nn as nn

from .models import ChristianModel, MagnusModel, SolveigModel


def load_model(modelname: str, *args, **kwargs) -> nn.Module:
    match modelname.lower():
        case "magnusmodel":
            return MagnusModel(*args, **kwargs)
        case "christianmodel":
            return ChristianModel(*args, **kwargs)
        case "solveigmodel":
            return SolveigModel(*args, **kwargs)
        case _:
            raise ValueError(
                f"Model: {modelname} has not been implemented. \nCheck the documentation for implemented metrics, or check your spelling"
            )
