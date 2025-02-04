from utils.dataloaders.usps_0_6 import USPSDataset0_6


def test_uspsdataset0_6():
    from pathlib import Path
    from tempfile import TemporaryFile

    import h5py
    import numpy as np

    with TemporaryFile() as tf:
        with h5py.File(tf, "w") as f:
            f["train/data"] = np.random.rand(10, 16 * 16)
            f["train/target"] = np.array([6, 5, 4, 3, 2, 1, 0, 0, 0, 0])

        dataset = USPSDataset0_6(data_path=tf, train=True)
        assert len(dataset) == 10
        data, target = dataset[0]
        assert data.shape == (1, 16, 16)
        assert all(target == np.array([0, 0, 0, 0, 0, 0, 1]))
