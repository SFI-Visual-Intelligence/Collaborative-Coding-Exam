import torch as th
import torch.nn as nn 

import argparse
from utils import load_metric, load_model, createfolders












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
    
    #Training specific values
    parser.add_argument('--epoch', type=int, default=20, help='Amount of training epochs the model will do.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate parameter for model training.')
    args = parser.parse_args()

    createfolders(args)
    
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    
    #load model
    model = load_model()
    model.to(device)
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr = args.learning_rate)



    for epoch in range(args.epoch):
        
        #Training loop start
        
        #Eval loop start
    
        pass 


if __name__ == '__main__':
    main()