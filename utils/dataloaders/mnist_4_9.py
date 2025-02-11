import gzip
import os
import urllib.request as ur
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

class MNIST_4_9(Dataset):
    def __init__(self,
                 datapath: Path,
                 train: bool = False,
                 download: bool = False
    ):
        super.__init__()
        self.datapath = datapath
        self.mnist_path = self.datapath / "MNIST"
        self.train = train
        self.download = download
        self.num_classes: int = 6
        
        if not self.download and not self._already_downloaded():
            raise FileNotFoundError(
                'Data files are not found. Set --download-data=True to download the data'
            )
        if self.download and not self._already_downloaded():
            self._download()
            
        
        
    
    def _download(self):
        urls: dict = {
            "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
            "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
            "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
        }
        
        
        for url in urls.values():
            file_path: Path = os.path.join(self.mnist_path, url.split('/')[-1])
            file_name: Path = file_path.replace('.gz','')
            if os.path.exists(file_name):
                print(f"File: {file_name} already downloaded")
            else:
                print(f"File: {file_name} is downloading...")
                ur.urlretrieve(url, file_path) # Download file
                with gzip.open(file_path, 'rb') as infile:
                    with open(file_name, 'wb') as outfile:
                        outfile.write(infile.read()) # Write from url to local file
                    os.remove(file_path) # remove .gz file
                    
                    
    
    def _already_downloaded(self):
        if self.mnist_path.exists():
            required_files: list = [
                "train-images-idx3-ubyte",
                "train-labels-idx1-ubyte",
                "t10k-images-idx3-ubyte",
                "t10k-labels-idx1-ubyte",
            ]
            return all([(self.mnist_path / file).exists() for file in required_files])

        else:
            self.mnist_path.mkdir(parents=True, exist_ok=True)
            return False
    
    def __len__(self):
        pass
    
    def __getitem__(self):
        pass
    
    