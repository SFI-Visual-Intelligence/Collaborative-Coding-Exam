from utils.dataloaders.usps_0_6 import USPSDataset0_6


def test_uspsdataset0_6():
    from pathlib import Path

    import numpy as np

    datapath = Path("data/USPS")

    dataset = USPSDataset0_6(data_path=datapath, train=True)
    assert len(dataset) == 5460
    data, target = dataset[0]
    assert data.shape == (1, 16, 16)
    assert all(target == np.array([0, 0, 0, 0, 0, 0, 1]))
