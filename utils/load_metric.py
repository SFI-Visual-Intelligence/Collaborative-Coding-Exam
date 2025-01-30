import copy 
import numpy as np 
import torch.nn as nn 
from metrics import EntropyPrediction


class MetricWrapper(nn.Module):
    def __init__(self,
                 EntropyPred:bool = True, 
                 F1Score:bool = True,
                 Recall:bool = True,
                 Precision:bool = True,
                 Accuracy:bool = True):
        super().__init__()
        self.metrics = {}
        
        if EntropyPred:
            self.metrics['Entropy of Predictions'] = EntropyPrediction()

        if F1Score:
            self.metrics['F1 Score'] = None 
        
        if Recall:
            self.metrics['Recall'] = None 
        
        if Precision:
            self.metrics['Precision'] = None
        
        if Accuracy:
            self.metrics['Accuracy'] = None
            
        self.tmp_scores = copy.deepcopy(self.metrics)
        for key in self.tmp_scores:
            self.tmp_scores[key] = []

    def __call__(self, y_true, y_pred):
        for key in self.metrics:
            self.tmp_scores[key].append(self.metrics[key](y_true, y_pred))
        
    def __getmetrics__(self):
        return_metrics = {}
        for key in self.metrics:
            return_metrics[key] = np.mean(self.tmp_scores[key])
        
        return return_metrics

    def __resetvalues__(self):
        for key in self.tmp_scores:
            self.tmp_scores[key] = []
