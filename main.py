import torch as th
import torch.nn as nn 

import argparse
from utils import load_metric












def main():
    
    parser = argparse.ArgumentParser(
        prog='',
        description='',
        epilog='',
    )
    
    parser.add_argument('--epoch', type=int, default=20, help='Amount of training epochs the model will do')
    
    args = parser.parse_args()




    for epoch in range(args.epoch):
        
        #Training loop start
        
        #Eval loop start
    



if __name__ == '__main__':
    main()