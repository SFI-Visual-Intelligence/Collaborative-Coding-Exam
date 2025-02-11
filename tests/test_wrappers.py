from utils import load_data, load_metric, load_model


def test_load_model():
    import torch as th

    image_shape = (1, 28, 28)
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
    pass
