"""This module contains the data sources for the datasets used in the experiments.

The data sources are defined as dictionaries with the following keys
- train: A list containing the URL, filename, and MD5 hash of the training data.
- test: A list containing the URL, filename, and MD5 hash of the test data.
"""

USPS_SOURCE = {
    "train": [
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
        "usps.bz2",
        "ec16c51db3855ca6c91edd34d0e9b197",
    ],
    "test": [
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
        "usps.t.bz2",
        "8ea070ee2aca1ac39742fdd1ef5ed118",
    ],
}

MNIST_SOURCE = {
    "train_images": ["https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", 
                     "train-images-idx3-ubyte", 
                     None
    ],
    "train_labels": ["https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
                    "train-labels-idx1-ubyte",
                    None
    ],
    "test_images": ["https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
                    "t10k-images-idx3-ubyte",
                    None
    ],
    "test_labels": ["https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
                    "t10k-labels-idx1-ubyte",
                    None
    ],
}
