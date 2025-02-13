import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """
    CNN block with Conv2d, MaxPool2d, and ReLU.

    Args
    ----

    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.relu(x)
        return x


def find_fc_input_shape(image_shape, *cnn_layers):
    """
    Find the shape of the input to the fully connected layer.

    Code inspired by @Seilmast (https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam/issues/67#issuecomment-2651212254)

    Args
    ----
    image_shape : tuple(int, int, int)
        Shape of the input image (C, H, W).
    cnn_layers : nn.Module
        List of CNN layers.

    Returns
    -------
    int
        Number of elements in the input to the fully connected layer.
    """

    dummy_img = torch.randn(1, *image_shape)
    with torch.no_grad():
        x = cnn_layers[0](dummy_img)

        for layer in cnn_layers[1:]:
            x = layer(x)

        x = x.view(x.size(0), -1)

    return x.size(1)


class ChristianModel(nn.Module):
    """Simple CNN model for image classification.

    Args
    ----
    image_shape : tuple(int, int, int)
        Shape of the input image (C, H, W).
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

    Input Image Shape: (5, 1, 16, 16)
    CNN1 Output Shape: (5, 50, 8, 8)
    CNN2 Output Shape: (5, 100, 4, 4)
    FC Output Shape: (5, num_classes)
    """

    def __init__(self, image_shape, num_classes):
        super().__init__()

        C, *_ = image_shape

        self.cnn1 = CNNBlock(C, 50)
        self.cnn2 = CNNBlock(50, 100)

        fc_input_shape = find_fc_input_shape(image_shape, self.cnn1, self.cnn2)

        self.fc1 = nn.Linear(fc_input_shape, num_classes)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    x = torch.randn(3, 3, 28, 28)

    model = ChristianModel(x.shape[1:], 7)

    y = model(x)

    print(y)
