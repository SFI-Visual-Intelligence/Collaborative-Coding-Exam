import torch.nn as nn


class MagnusModel(nn.Module):
    def __init__(self,
                 imgsize: int,
                 channels: int,
                 n_classes:int=10):
        super().__init__()
        self.imgsize = imgsize 
        self.channels = channels
        
        self.layer1 = nn.Sequential(*([
            nn.Linear(self.channels*self.imgsize*self.imgsize, 133),
            nn.ReLU()
        ]))
        self.layer2 = nn.Sequential(*([
            nn.Linear(133, 133),
            nn.ReLU()
        ]))
        self.layer3 = nn.Sequential(*([
            nn.Linear(133, n_classes),
            nn.ReLU()
        ]))

    def forward(self, x):
        assert len(x.size) == 4
        
        x = x.view(x.size(0), -1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
    
        return x
