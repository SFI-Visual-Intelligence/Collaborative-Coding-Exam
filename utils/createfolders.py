import argparse
import os
from tempfile import TemporaryDirectory


def createfolders(args) -> None:
    """
    Creates folders for storing data, results, model weights.

    Parameters
    ----------
    args
        ArgParse object containing string paths to be created

    """

    if not os.path.exists(args.datafolder):
        os.makedirs(args.datafolder)
        print(f"Created a folder at {args.datafolder}")

    if not os.path.exists(args.resultfolder):
        os.makedirs(args.resultfolder)
        print(f"Created a folder at {args.resultfolder}")

    if not os.path.exists(args.modelfolder):
        os.makedirs(args.modelfolder)
        print(f"Created a folder at {args.modelfolder}")


def test_createfolders():
    with TemporaryDirectory(dir="tmp/") as temp_dir:
        parser = argparse.ArgumentParser()
        # Structuture related values
        parser.add_argument(
            "--datafolder",
            type=str,
            default=os.path.join(temp_dir, "Data/"),
            help="Path to where data will be saved during training.",
        )
        parser.add_argument(
            "--resultfolder",
            type=str,
            default=os.path.join(temp_dir, "Results/"),
            help="Path to where results will be saved during evaluation.",
        )
        parser.add_argument(
            "--modelfolder",
            type=str,
            default=os.path.join(temp_dir, "Experiments/"),
            help="Path to where model weights will be saved at the end of training.",
        )

        args = parser.parse_args()
        createfolders(args)

    return
