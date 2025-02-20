import numpy as np
import torch as th
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from CollaborativeCoding import (
    MetricWrapper,
    createfolders,
    get_args,
    load_data,
    load_model,
)

# from wandb_api import WANDB_API


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

    if "usps" in args.dataset.lower():
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
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
        nr_channels=args.nr_channels,
    )

    train_metrics = MetricWrapper(
        *args.metric,
        num_classes=traindata.num_classes,
        macro_averaging=args.macro_averaging,
    )
    val_metrics = MetricWrapper(
        *args.metric,
        num_classes=traindata.num_classes,
        macro_averaging=args.macro_averaging,
    )
    test_metrics = MetricWrapper(
        *args.metric,
        num_classes=traindata.num_classes,
        macro_averaging=args.macro_averaging,
    )

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

            train_metrics(y, logits)

            break
        print(train_metrics.getmetrics())
        print("Dry run completed successfully.")
        exit()

    #    wandb.login(key=WANDB_API)
    wandb.init(
        entity="ColabCode",
        project=args.run_name,
        tags=[args.modelname, args.dataset],
        config=args,
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

            train_metrics(y, logits)

        valloss = []
        # Validation loop start
        model.eval()
        with th.no_grad():
            for x, y in tqdm(valiloader, desc="Validation"):
                x, y = x.to(device), y.to(device)
                logits = model.forward(x)
                loss = criterion(logits, y)
                valloss.append(loss.item())

                val_metrics(y, logits)

        wandb.log(
            {
                "Epoch": epoch,
                "Train loss": np.mean(trainingloss),
                "Validation loss": np.mean(valloss),
            }
            | train_metrics.getmetric(str_prefix="Train ")
            | val_metrics.getmetric(str_prefix="Validation ")
        )
        train_metrics.resetmetric()
        val_metrics.resetmetric()

    testloss = []
    model.eval()
    with th.no_grad():
        for x, y in tqdm(testloader, desc="Testing"):
            x, y = x.to(device), y.to(device)
            logits = model.forward(x)
            loss = criterion(logits, y)
            testloss.append(loss.item())

            preds = th.argmax(logits, dim=1)
            test_metrics(y, preds)

    wandb.log(
        {"Epoch": 1, "Test loss": np.mean(testloss)}
        | test_metrics.getmetric(str_prefix="Test ")
    )
    test_metrics.resetmetric()


if __name__ == "__main__":
    main()
