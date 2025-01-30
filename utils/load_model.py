import torch.nn as nn 

def load_model(modelname:str) -> nn.Module:
    
    raise ValueError(f'Metric: {modelname} has not been implemented. \nCheck the documentation for implemented metrics, or check your spelling')