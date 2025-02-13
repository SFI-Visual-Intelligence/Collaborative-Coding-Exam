from CollaborativeCoding.dataloaders.usps_0_6 import USPSDataset0_6


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
            targets = np.array([6, 5, 4, 3, 2, 1, 0, 0, 0, 0])
            indices = np.arange(len(targets))
            # Populate the file with data
            f["train/data"] = np.random.rand(10, 16 * 16)
            f["train/target"] = targets

        trans = transforms.Compose(
            [
                transforms.Resize((16, 16)),
                transforms.ToTensor(),
            ]
        )
        dataset = USPSDataset0_6(
            data_path=tempdir,
            sample_ids=indices,
            train=True,
            transform=trans,
        )
        assert len(dataset) == 10
        data, target = dataset[0]
        assert data.shape == (1, 16, 16)
        assert target == 6
