from utils.dataloaders import SVHNDataset, USPSDataset0_6


def test_uspsdataset0_6():
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import h5py
    import numpy as np
    from torchvision import transforms

    # Create a temporary directory (deleted after the test)
    with TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)

        tf = tempdir / "usps.h5"

        # Create a h5 file
        with h5py.File(tf, "w") as f:
            # Populate the file with data
            f["train/data"] = np.random.rand(10, 16 * 16)
            f["train/target"] = np.array([6, 5, 4, 3, 2, 1, 0, 0, 0, 0])

        trans = transforms.Compose(
            [
                transforms.Resize((16, 16)),  # At least for USPS
                transforms.ToTensor(),
            ]
        )
        dataset = USPSDataset0_6(data_path=tempdir, train=True, transform=trans)
        assert len(dataset) == 10
        data, target = dataset[0]
        assert data.shape == (1, 16, 16)
        assert all(target == np.array([0, 0, 0, 0, 0, 0, 1]))


def test_svhn_dataset():
    import os
    from tempfile import TemporaryDirectory

    from torchvision import transforms

    with TemporaryDirectory() as tempdir:
        trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

        dataset = SVHNDataset(
            tempdir, train=True, transform=trans, download=True, nr_channels=1
        )

        assert dataset.__len__() != 0
        assert os.path.exists(os.path.join(tempdir, "train_32x32.mat"))

        img, label = dataset.__getitem__(0)
        assert len(img.size()) == 3 and img.size() == (1, 28, 28) and img.size(0) == 1
        assert len(label.size()) == 1
