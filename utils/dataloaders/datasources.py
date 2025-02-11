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
