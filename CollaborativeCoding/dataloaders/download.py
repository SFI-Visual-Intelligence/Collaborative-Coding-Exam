import bz2
import gzip
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

import h5py as h5
import numpy as np
from scipy.io import loadmat
from torchvision.datasets import SVHN

from .datasources import MNIST_SOURCE, USPS_SOURCE


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
    >>> from CollaborativeCoding import Downloader
    >>> dir = Path('tmp')
    >>> dir.mkdir(exist_ok=True)
    >>> train, test = Downloader().usps(dir)
    """

    def mnist(self, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        def _chech_is_downloaded(path: Path) -> bool:
            path = path / "MNIST"
            if path.exists():
                required_files = [MNIST_SOURCE[key][1] for key in MNIST_SOURCE.keys()]
                if all([(path / file).exists() for file in required_files]):
                    print("MNIST Dataset already downloaded.")
                    return True
                else:
                    return False
            else:
                path.mkdir(parents=True, exist_ok=True)
                return False

        def _download_data(path: Path) -> None:
            path = path / "MNIST"
            urls = {key: MNIST_SOURCE[key][0] for key in MNIST_SOURCE.keys()}

            for name, url in urls.items():
                file_path = os.path.join(path, url.split("/")[-1])
                if not os.path.exists(
                    file_path.replace(".gz", "")
                ):  # Avoid re-downloading
                    urlretrieve(url, file_path, reporthook=self.__reporthook)
                    with gzip.open(file_path, "rb") as f_in:
                        with open(file_path.replace(".gz", ""), "wb") as f_out:
                            f_out.write(f_in.read())
                    os.remove(file_path)  # Remove compressed file

        def _get_labels(path: Path) -> np.ndarray:
            with open(path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
            return data

        if not _chech_is_downloaded(data_dir):
            _download_data(data_dir)

        train_labels_path = data_dir / "MNIST" / MNIST_SOURCE["train_labels"][1]
        test_labels_path = data_dir / "MNIST" / MNIST_SOURCE["test_labels"][1]

        train_labels = _get_labels(train_labels_path)
        test_labels = _get_labels(test_labels_path)

        return train_labels, test_labels

    def svhn(self, data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        def download_svhn(path, train: bool = True):
            SVHN(path, split="train" if train else "test", download=True)

        parent_path = data_dir / "SVHN"

        if not parent_path.exists():
            parent_path.mkdir(parents=True)

        train_data = parent_path / "train_32x32.mat"
        test_data = parent_path / "test_32x32.mat"

        if not train_data.is_file():
            download_svhn(parent_path, train=True)
        if not test_data.is_file():
            download_svhn(parent_path, train=False)
        print(test_data)
        train_labels = loadmat(train_data)["y"]
        test_labels = loadmat(test_data)["y"]

        return train_labels, test_labels

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
        with TemporaryDirectory(dir=data_dir) as tmp_dir:
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
