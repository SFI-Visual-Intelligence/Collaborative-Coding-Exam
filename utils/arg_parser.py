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

    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Whether the data should be downloaded or not. Might cause code to start a bit slowly.",
    )

    # Data/Model specific values
    parser.add_argument(
        "--modelname",
        type=str,
        default="MagnusModel",
        choices=["MagnusModel", "ChristianModel", "SolveigModel", "JanModel", "JohanModel"],
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
        default="cpu",
        choices=["cuda", "cpu", "mps"],
        help="Which device to run the training on.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If true, the code will not run the training loop.",
    )
    args = parser.parse_args()
    
    
    assert args.epoch > 0, "Epoch should be a positive integer."
    assert args.learning_rate > 0, "Learning rate should be a positive float."
    assert args.batchsize > 0, "Batch size should be a positive integer."
   
    
    
    return args
