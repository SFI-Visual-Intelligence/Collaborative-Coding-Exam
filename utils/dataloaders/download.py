import bz2
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import h5py as h5
import numpy as np

from .datasources import USPS_SOURCE


class Downloader:
    """
    Class to download and load the USPS dataset.

    Methods
    -------
    mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray]
        Download the MNIST dataset and save it as an HDF5 file to `data_dir`.
    svhn(data_dir: Path) -> tuple[np.ndarray, np.ndarray]
        Download the SVHN dataset and save it as an HDF5 file to `data_dir`.
    usps(data_dir: Path) -> tuple[np.ndarray, np.ndarray]
        Download the USPS dataset and save it as an HDF5 file to `data_dir`.

    Raises
    ------
    NotImplementedError
        If the download method is not implemented for the dataset.

    Examples
    --------
    >>> from pathlib import Path
    >>> from utils import Downloader
    >>> dir = Path('tmp')
    >>> dir.mkdir(exist_ok=True)
    >>> train, test = Downloader().usps(dir)
    """

    def mnist(self, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("MNIST download not implemented yet")

    def svhn(self, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("SVHN download not implemented yet")

    def usps(self, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Download the USPS dataset and save it as an HDF5 file to `data_dir/usps.h5`.
        """

        def already_downloaded(path):
            if not path.exists() or not path.is_file():
                return False

            with h5.File(path, "r") as f:
                return "train" in f and "test" in f

        filename = data_dir / "usps.h5"

        if already_downloaded(filename):
            with h5.File(filename, "r") as f:
                return f["train"]["target"][:], f["test"]["target"][:]

        url_train, _, train_md5 = USPS_SOURCE["train"]
        url_test, _, test_md5 = USPS_SOURCE["test"]

        # Using temporary directory ensures temporary files are deleted after use
        with TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / "train"
            test_path = Path(tmp_dir) / "test"

            # Download the dataset and report the progress
            urlretrieve(url_train, train_path, reporthook=self.__reporthook)
            self.__check_integrity(train_path, train_md5)
            train_targets = self.__extract_usps(train_path, filename, "train")

            urlretrieve(url_test, test_path, reporthook=self.__reporthook)
            self.__check_integrity(test_path, test_md5)
            test_targets = self.__extract_usps(test_path, filename, "test")

        return train_targets, test_targets

    def __extract_usps(self, src: Path, dest: Path, mode: str):
        # Load the dataset and save it as an HDF5 file
        with bz2.open(src) as fp:
            raw = [line.decode().split() for line in fp.readlines()]

            tmp = [[x.split(":")[-1] for x in data[1:]] for data in raw]

            imgs = np.asarray(tmp, dtype=np.float32)
            imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)

            targets = [int(d[0]) - 1 for d in raw]

        with h5.File(dest, "a") as f:
            f.create_dataset(f"{mode}/data", data=imgs, dtype=np.float32)
            f.create_dataset(f"{mode}/target", data=targets, dtype=np.int32)

        return targets

    @staticmethod
    def __reporthook(blocknum, blocksize, totalsize):
        """
        Use this function to report download progress
        for the urllib.request.urlretrieve function.
        """

        denom = 1024 * 1024
        readsofar = blocknum * blocksize

        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = f"\r{int(percent):^3}% {readsofar / denom:.2f} of {totalsize / denom:.2f} MB"
            print(s, end="", flush=True)
            if readsofar >= totalsize:
                print()

    @staticmethod
    def __check_integrity(filepath, checksum):
        """Check the integrity of the USPS dataset file.

        Args
        ----
        filepath : pathlib.Path
            Path to the USPS dataset file.
        checksum : str
            MD5 checksum of the dataset file.

        Returns
        -------
        bool
            True if the checksum of the file matches the expected checksum, False otherwise
        """

        file_hash = hashlib.md5(filepath.read_bytes()).hexdigest()

        if not checksum == file_hash:
            raise ValueError(
                f"File integrity check failed. Expected {checksum}, got {file_hash}"
            )
