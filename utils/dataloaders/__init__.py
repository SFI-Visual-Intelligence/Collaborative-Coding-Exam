__all__ = [
    "USPSDataset0_6",
    "USPSH5_Digit_7_9_Dataset",
    "MNISTDataset0_3",
    "MNISTDataset4_9",
    "Downloader",
    "SVHNDataset",
]

from .download import Downloader
from .mnist_0_3 import MNISTDataset0_3
from .mnist_4_9 import MNISTDataset4_9
from .svhn import SVHNDataset
from .usps_0_6 import USPSDataset0_6
from .uspsh5_7_9 import USPSH5_Digit_7_9_Dataset
