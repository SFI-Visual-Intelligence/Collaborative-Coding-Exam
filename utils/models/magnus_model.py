import torch.nn as nn


class MagnusModel(nn.Module):
    def __init__(self, image_shape, num_classes: int, nr_channels: int):
        """
        Initializes the MagnusModel, a neural network designed for image classification tasks.

        The model consists of three linear layers, each with 133 neurons, and uses ReLU activation
        functions between the layers. The first layer's input size is determined by the image shape
        and number of channels, while the output layer's size is determined by the number of classes.
        Args:
            image_shape (tuple): A tuple representing the dimensions of the input image (Channels, Height, Width).
            num_classes (int): The number of output classes for classification.
            nr_channels (int): The number of channels in the input image.
        Returns:
            MagnusModel (nn.Module): An instance of the MagnusModel neural network.
        """
        super().__init__()
        *_, H, W = image_shape

        self.layer1 = nn.Sequential(
            *(
                [
                    nn.Linear(nr_channels * H * W, 133),
                    nn.ReLU(),
                ]
            )
        )
        self.layer2 = nn.Sequential(*([nn.Linear(133, 133), nn.ReLU()]))
        self.layer3 = nn.Sequential(
            *(
                [
                    nn.Linear(133, num_classes),
                ]
            )
        )

    def forward(self, x):
        """
        Defines the forward pass of the MagnusModel.
        Args:
            x (torch.Tensor): A four-dimensional tensor with shape (Batch Size, Channels, Image Height, Image Width).
        Returns:
            torch.Tensor: The output tensor containing class logits for each input sample.
        """
        assert len(x.size()) == 4
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        return out


if __name__ == "__main__":
    import torch as th

    image_shape = (3, 28, 28)
    n, c, h, w = 5, *image_shape
    model = MagnusModel([h, w], 10, c)

    x = th.rand((n, c, h, w))
    with th.no_grad():
        y = model(x)

    assert y.shape == (n, 10), f"Shape: {y.shape}"
