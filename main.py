import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import wandb
from utils import MetricWrapper, createfolders, get_args, load_data, load_model
from wandb_api import WANDB_API

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

    if args.dataset.lower() in ["usps_0-6", "usps_7-9"]:
        transform = transforms.Compose(
            [
                transforms.Resize((16, 16)),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    traindata, validata, testdata = load_data(
        args.dataset,
        data_dir=args.datafolder,
        transform=transform,
        val_size=args.val_size,
    )

    metrics = MetricWrapper(*args.metric, num_classes=traindata.num_classes, macro_averaging=args.macro_averaging)

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
    testloader = DataLoader(
        testdata, batch_size=args.batchsize, shuffle=False, pin_memory=True
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
        exit()

    # wandb.login(key=WANDB_API)
    wandb.init(
            entity="ColabCode",
            # entity="FYS-8805 Exam",
            project="Jan", 
            tags=[args.modelname, args.dataset]
            )
    wandb.watch(model)
    
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

        valloss = []
        # Validation loop start
        model.eval()
        with th.no_grad():
            for x, y in tqdm(valiloader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                logits = model.forward(x)
                loss = criterion(logits, y)
                valloss.append(loss.item())

                metrics(y, logits)

        wandb.log(metrics.accumulate(str_prefix="Validation "))
        metrics.reset()

        wandb.log(
            {
                "Epoch": epoch,
                "Train loss": np.mean(trainingloss),
                "Validation loss": np.mean(valloss),
            }
        )

    testloss = []
    model.eval()
    with th.no_grad():
        for x, y in tqdm(testloader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            logits = model.forward(x)
            loss = criterion(logits, y)
            testloss.append(loss.item())

            preds = th.argmax(logits, dim=1)
            metrics(y, preds)

    wandb.log(metrics.accumulate(str_prefix="Test "))
    metrics.reset()
    wandb.log({"Test loss": np.mean(testloss)})


if __name__ == "__main__":
    main()
