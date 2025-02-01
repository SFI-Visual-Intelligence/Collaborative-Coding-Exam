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
