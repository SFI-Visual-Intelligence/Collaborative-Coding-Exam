from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import MetricWrapper, createfolders, get_args, load_data, load_model


def main():
    """

    Parameters
    ----------

    Returns
    -------

    Raises
    ------

    """

    args = get_args()


    createfolders(args.datafolder, args.resultfolder, args.modelfolder)

    device = args.device

    if args.dataset.lower() in ["usps_0-6", "uspsh5_7_9"]:
        augmentations = transforms.Compose(
            [
                transforms.Resize((16, 16)),
                transforms.ToTensor(),
            ]
        )
    else:
        augmentations = transforms.Compose([transforms.ToTensor()])

    # Dataset
    traindata = load_data(
        args.dataset,
        train=True,
        data_path=args.datafolder,
        download=args.download_data,
        transform=augmentations,
    )
    validata = load_data(
        args.dataset,
        train=False,
        data_path=args.datafolder,
        download=args.download_data,
        transform=augmentations,
    )

    metrics = MetricWrapper(*args.metric, num_classes=traindata.num_classes)

    # Find the shape of the data, if is 2D, add a channel dimension
    data_shape = traindata[0][0].shape
    if len(data_shape) == 2:
        data_shape = (1, *data_shape)

    # load model
    model = load_model(
        args.modelname,
        image_shape=data_shape,
        num_classes=traindata.num_classes,
    )
    model.to(device)

    trainloader = DataLoader(
        traindata,
        batch_size=args.batchsize,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valiloader = DataLoader(
        validata, batch_size=args.batchsize, shuffle=False, pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)

    # This allows us to load all the components without running the training loop
    if args.dry_run:
        dry_run_loader = DataLoader(
            traindata,
            batch_size=20,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        for x, y in tqdm(dry_run_loader, desc="Dry run", total=1):
            x, y = x.to(device), y.to(device)
            logits = model.forward(x)

            loss = criterion(logits, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            metrics(y, logits)

            break
        print(metrics.accumulate())
        print("Dry run completed successfully.")
        exit(0)

    # wandb.login(key=WANDB_API)
    wandb.init(
            entity="ColabCode-org",
            # entity="FYS-8805 Exam",
            project="Test", 
            tags=[args.modelname, args.dataset]
            )
    wandb.watch(model)
    exit()
    for epoch in range(args.epoch):
        # Training loop start
        trainingloss = []
        model.train()
        for x, y in tqdm(trainloader, desc="Training"):
            x, y = x.to(device), y.to(device)
            logits = model.forward(x)

            loss = criterion(logits, y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            trainingloss.append(loss.item())

            metrics(y, logits)

        wandb.log(metrics.accumulate(str_prefix="Train "))
        metrics.reset()

        evalloss = []
        # Eval loop start
        model.eval()
        with th.no_grad():
            for x, y in tqdm(valiloader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                logits = model.forward(x)
                loss = criterion(logits, y)
                evalloss.append(loss.item())

                metrics(y, logits)

        wandb.log(metrics.accumulate(str_prefix="Evaluation "))
        metrics.reset()

        wandb.log(
            {
                "Epoch": epoch,
                "Train loss": np.mean(trainingloss),
                "Evaluation Loss": np.mean(evalloss),
            }
        )


if __name__ == "__main__":
    main()
