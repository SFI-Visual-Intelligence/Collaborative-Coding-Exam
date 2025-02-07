import os

from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.datasets import SVHN


class SVHNDataset(Dataset):
    def __init__(
        self, datapath: str, transforms=None, download_data=True, split="train"
    ):
        """
        Initializes the SVHNDataset object.
        Args:
            datapath (str): Path to where the data lies. If download_data is set to True, this is where the data will be downloaded.
            transforms: Torch composite of transformations which are to be applied to the dataset images.
            download_data (bool): If True, downloads the dataset to the specified datapath.
            split (str): The dataset split to use, either 'train' or 'test'.
        Raises:
            AssertionError: If the split is not 'train' or 'test'.
        """
        super().__init__()
        assert split == "train" or split == "test"

        if download_data:
            self._download_data(datapath, split)

        data = loadmat(os.path.join(datapath, f"{split}_32x32.mat"))

        # Images on the form N x H x W x C
        self.images = data["X"].transpose(3, 1, 0, 2)
        self.labels = data["y"].flatten()
        self.labels[self.labels == 10] = 0

        self.transforms = transforms

    def _download_data(self, path: str, split: str):
        """
        Downloads the SVHN dataset.
        Args:
            path (str): The directory where the dataset will be downloaded.
            split (str): The dataset split to download, either 'train' or 'test'.
        """
        print(f"Downloading SVHN data into {path}")
        SVHN(path, split=split, download=True)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: The number of samples.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Retrieves the image and label at the specified index.
        Args:
            index (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        img, lab = self.images[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, lab
