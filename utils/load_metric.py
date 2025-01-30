import torch.nn as nn 
from metrics import EntropyPrediction

def load_metric(metricname:str) -> nn.Module:
    '''
    This function returns an instance of a class inhereting from nn.Module. 
    This class returns the given metric given a set of label - prediction pairs. 
    
    Parameters
    ----------
    metricname: string
        string naming the metric to return. 
    
    Returns
    -------
    Class
        Returns an instance of a class inhereting from nn.Module.
    
    Raises
    ------
    ValueError
        When the metricname parameter does not match any implemented metric, raise this error along with a descriptive message. 
    '''
    if metricname == 'EntropyPrediction':
        return EntropyPrediction()
    else:
        raise ValueError(f'Metric: {metricname} has not been implemented. \nCheck the documentation for implemented metrics, or check your spelling')