import argparse
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from utils import MetricWrapper, createfolders, load_data, load_model


def main():
    """

    Parameters
    ----------

    Returns
    -------

    Raises
    ------

    """
    parser = argparse.ArgumentParser(
        prog="",
        description="",
        epilog="",
    )
    # Structuture related values
    parser.add_argument(
        "--datafolder",
        type=Path,
        default="Data",
        help="Path to where data will be saved during training.",
    )
    parser.add_argument(
        "--resultfolder",
        type=Path,
        default="Results",
        help="Path to where results will be saved during evaluation.",
    )
    parser.add_argument(
        "--modelfolder",
        type=Path,
        default="Experiments",
        help="Path to where model weights will be saved at the end of training.",
    )
    parser.add_argument(
        "--savemodel",
        type=bool,
        default=False,
        help="Whether model should be saved or not.",
    )

    parser.add_argument(
        "--download-data",
        type=bool,
        default=False,
        help="Whether the data should be downloaded or not. Might cause code to start a bit slowly.",
    )

    # Data/Model specific values
    parser.add_argument(
        "--modelname",
        type=str,
        default="MagnusModel",
        choices=["MagnusModel"],
        help="Model which to be trained on",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="svhn",
        choices=["svhn", "usps_0-6"],
        help="Which dataset to train the model on.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default=["entropy"],
        choices=["entropy", "f1", "recall", "precision", "accuracy"],
        nargs="+",
        help="Which metric to use for evaluation",
    )

    # Training specific values
    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="Amount of training epochs the model will do.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate parameter for model training.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=64,
        help="Amount of training images loaded in one go",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Which device to run the training on.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If true, the code will not run the training loop.",
    )

    args = parser.parse_args()

    createfolders(args.datafolder, args.resultfolder, args.modelfolder)

    device = args.device

    # load model
    model = load_model(args.modelname)
    model.to(device)

    metrics = MetricWrapper(*args.metric)

    # Dataset
    traindata = load_data(
        args.dataset,
        train=True,
        data_path=args.datafolder,
        download=args.download_data,
    )
    validata = load_data(
        args.dataset,
        train=False,
        data_path=args.datafolder,
    )

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
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)

    # This allows us to load all the components without running the training loop
    if args.dry_run:
        print("Dry run completed")
        exit(0)

    wandb.init(project='',
               tags=[])
    wandb.watch(model)

    for epoch in range(args.epoch):

        # Training loop start
        trainingloss = []
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            pred = model.forward(x)

            loss = criterion(y, pred)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            trainingloss.append(loss.item())

        evalloss = []
        # Eval loop start
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
