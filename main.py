import torch as th
import torch.nn as nn 
from torch.utils.data import DataLoader
import argparse
import wandb
import numpy as np
from utils import MetricWrapper, load_model, load_data, createfolders


def main():
    '''
    
    Parameters
    ----------
    
    Returns
    -------
    
    Raises
    ------
    
    '''
    parser = argparse.ArgumentParser(
        prog='',
        description='',
        epilog='',
    )
    #Structuture related values
    parser.add_argument('--datafolder', type=str, default='Data/', help='Path to where data will be saved during training.')
    parser.add_argument('--resultfolder', type=str, default='Results/', help='Path to where results will be saved during evaluation.')
    parser.add_argument('--modelfolder', type=str, default='Experiments/', help='Path to where model weights will be saved at the end of training.')
    parser.add_argument('--savemodel', type=bool, default=False, help='Whether model should be saved or not.')
    
    parser.add_argument('--download-data', type=bool, default=False, help='Whether the data should be downloaded or not. Might cause code to start a bit slowly.')
    
    #Data/Model specific values
    parser.add_argument('--modelname', type=str, default='MagnusModel', 
                        choices = ['MagnusModel'], help="Model which to be trained on")
    parser.add_argument('--dataset', type=str, default='svhn',
                        choices=['svhn'], help='Which dataset to train the model on.')
    
    parser.add_argument("--metric", type=str, default="entropy", choices=['entropy', 'f1', 'recall', 'precision', 'accuracy'], nargs="+", help='Which metric to use for evaluation')

    #Training specific values
    parser.add_argument('--epoch', type=int, default=20, help='Amount of training epochs the model will do.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate parameter for model training.')
    parser.add_argument('--batchsize', type=int, default=64, help='Amount of training images loaded in one go')
    
    args = parser.parse_args()
    

    createfolders(args)
    
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    
    #load model
    model = load_model()
    model.to(device)
    
    metrics = MetricWrapper(*args.metric)
    
    #Dataset
    traindata = load_data(args.dataset)
    validata = load_data(args.dataset)
    
    trainloader = DataLoader(traindata,
                             batch_size=args.batchsize,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)
    valiloader = DataLoader(validata,
                            batch_size=args.batchsize,
                            shuffle=False,
                            pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr = args.learning_rate)
    
    
    wandb.init(project='',
               tags=[])
    wandb.watch(model)
    
    for epoch in range(args.epoch):
        
        #Training loop start
        trainingloss = []
        model.train()
        for x, y in traindata:
            x, y = x.to(device), y.to(device)
            pred = model.forward(x)
             
            loss = criterion(y, pred)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            trainingloss.append(loss.item())
        
        evalloss = []
        #Eval loop start
        model.eval()
        with th.no_grad():
            for x, y in valiloader:
                x = x.to(device)
                pred = model.forward(x)
                loss = criterion(y, pred)
                evalloss.append(loss.item())
                
        wandb.log({
            'Epoch': epoch,
            'Train loss': np.mean(trainingloss),
            'Evaluation Loss': np.mean(evalloss)
        })
              


if __name__ == '__main__':
    main()
