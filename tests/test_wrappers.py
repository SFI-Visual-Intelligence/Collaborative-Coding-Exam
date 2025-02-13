from CollaborativeCoding import load_data, load_metric, load_model


def test_load_model():
    import torch as th

    image_shape = (1, 16, 16)
    num_classes = 4

    dummy_img = th.rand((1, *image_shape))

    modelnames = [
        "magnusmodel",
        "christianmodel",
        "janmodel",
        "solveigmodel",
        "johanmodel",
    ]

    for name in modelnames:
        model = load_model(name, image_shape=image_shape, num_classes=num_classes)

        with th.no_grad():
            output = model(dummy_img)
            assert output.size() == (1, 4), (
                f"Model {name} returned image of size {output}. Expected (1,4)"
            )


def test_load_data():
    from tempfile import TemporaryDirectory

    import torch as th
    from torchvision import transforms

    dataset_names = [
        "usps_0-6",
        "mnist_0-3",
        "usps_7-9",
        "svhn",
        # 'mnist_4-9' #Uncomment when implemented
    ]

    trans = transforms.Compose(
        [
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ]
    )

    with TemporaryDirectory() as tmppath:
        for name in dataset_names:
            dataset = load_data(
                name, train=False, data_path=tmppath, download=True, transform=trans
            )

            im, _ = dataset.__getitem__(0)

            assert dataset.__len__() != 0
            assert type(im) == th.Tensor and len(im.size()) == 3


def test_load_metric():
    pass
