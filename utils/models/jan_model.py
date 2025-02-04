import torch
import torch.nn as nn


class JanModel(nn.Module):
    """A simple MLP network model for image classification tasks.

    Args
    ----
    in_channels : int
        Number of input channels.
    num_classes : int
        Number of classes in the dataset.

    Processing Images
    -----------------
    Input: (N, C, H, W)
        N: Batch size
        C: Number of input channels
        H: Height of the input image
        W: Width of the input image

    Example:
    For grayscale images, C = 1.

    Input Image Shape: (5, 1, 28, 28)
    flatten Output Shape: (5, 784)
    fc1 Output Shape: (5, 100)
    fc2 Output Shape: (5, 100)
    out Output Shape: (5, num_classes)
    """

    def __init__(self, image_shape, num_classes):
        super().__init__()

        self.in_channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.height * self.width * self.in_channels, 100)

        self.fc2 = nn.Linear(100, 100)

        self.out = nn.Linear(100, num_classes)

        self.leaky_relu = nn.LeakyReLU()

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    model = JanModel(2, 4)

    x = torch.randn(3, 2, 28, 28)
    y = model(x)

    print(y)
