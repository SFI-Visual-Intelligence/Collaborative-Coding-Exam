import torch.nn as nn

from .models import ChristianModel, JanModel, MagnusModel, SolveigModel


def load_model(modelname: str, *args, **kwargs) -> nn.Module:
    """
    Load the model based on the model name.

    Args
    ----
    modelname : str
        Name of the model to load.
    *args : list
        Additional arguments for the model class.
    **kwargs : dict
        Additional keyword arguments for the model class.

    Returns
    -------
    model : torch.nn.Module
        Model object.

    Raises
    ------
    NotImplementedError
        If the model is not implemented.

    Examples
    --------
    >>> from utils import load_model
    >>> model = load_model("magnusmodel", num_classes=10)
    >>> model
    MagnusModel(
      (fc1): Linear(in_features=784, out_features=100, bias=True)
      (fc2): Linear(in_features=100, out_features=10, bias=True
    """
    match modelname.lower():
        case "magnusmodel":
            return MagnusModel(*args, **kwargs)
        case "christianmodel":
            return ChristianModel(*args, **kwargs)
        case "janmodel":
            return JanModel(*args, **kwargs)
        case "solveigmodel":
            return SolveigModel(*args, **kwargs)
        case _:
            errmsg = (
                f"Model: {modelname} not implemented. "
                "Check the documentation for implemented models, "
                "or check your spelling."
            )
            raise NotImplementedError(errmsg)
