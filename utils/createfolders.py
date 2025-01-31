import argparse
from pathlib import Path
from tempfile import TemporaryDirectory


def createfolders(*dirs: Path) -> None:
    """
    Creates folders for storing data, results, model weights.

    Parameters
    ----------
    args
        ArgParse object containing string paths to be created

    """

    for dir in dirs:
        dir.mkdir(parents=True, exist_ok=True)


def test_createfolders():
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        parser = argparse.ArgumentParser()

        # Structuture related values
        parser.add_argument(
            "--datafolder",
            type=Path,
            default=temp_dir / "Data",
            help="Path to where data will be saved during training.",
        )
        parser.add_argument(
            "--resultfolder",
            type=Path,
            default=temp_dir / "Results",
            help="Path to where results will be saved during evaluation.",
        )
        parser.add_argument(
            "--modelfolder",
            type=Path,
            default=temp_dir / "Experiments",
            help="Path to where model weights will be saved at the end of training.",
        )

        args = parser.parse_args(
            [
                "--datafolder",
                temp_dir / "Data",
                "--resultfolder",
                temp_dir / "Results",
                "--modelfolder",
                temp_dir / "Experiments",
            ]
        )

        createfolders(args.datafolder, args.resultfolder, args.modelfolder)

        assert (temp_dir / "Data").exists()
        assert (temp_dir / "Results").exists()
        assert (temp_dir / "Experiments").exists()
