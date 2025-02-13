import argparse
from pathlib import Path


def get_args():
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
        action="store_true",
        help="Whether model should be saved or not.",
    )

    # Data/Model specific values
    parser.add_argument(
        "--modelname",
        type=str,
        default="MagnusModel",
        choices=[
            "MagnusModel",
            "ChristianModel",
            "SolveigModel",
            "JanModel",
            "JohanModel",
        ],
        help="Model which to be trained on",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="svhn",
        choices=["svhn", "usps_0-6", "usps_7-9", "mnist_0-3", "mnist_4-9"],
        help="Which dataset to train the model on.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Percentage of training dataset to be used as validation dataset - must be within (0,1).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=["entropy"],
        choices=["entropy", "f1", "recall", "precision", "accuracy"],
        nargs="+",
        help="Which metric to use for evaluation",
    )

    parser.add_argument("--imagesize", type=int, default=28, help="Imagesize")

    parser.add_argument(
        "--nr_channels",
        type=int,
        default=1,
        choices=[1, 3],
        help="Number of image channels",
    )
    parser.add_argument(
        "--macro_averaging",
        action="store_true",
        help="If the flag is included, the metrics will be calculated using macro averaging.",
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
        default="cpu",
        choices=["cuda", "cpu", "mps"],
        help="Which device to run the training on.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If the flag is included, the code will not run the training loop.",
    )

    parser.add_argument(
        "--run_name", type=str, required=True, help="Name for WANDB project"
    )
    args = parser.parse_args()

    assert args.epoch > 0, "Epoch should be a positive integer."
    assert args.learning_rate > 0, "Learning rate should be a positive float."
    assert args.batchsize > 0, "Batch size should be a positive integer."

    return args
