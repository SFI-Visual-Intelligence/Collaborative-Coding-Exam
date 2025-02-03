import torch
import torch.nn as nn


class CNNBlock(nn.Module):
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


class ChristianModel(nn.Module):
    """Simple CNN model for image classification.

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

    Input Image Shape: (5, 1, 16, 16)
    CNN1 Output Shape: (5, 50, 8, 8)
    CNN2 Output Shape: (5, 100, 4, 4)
    FC Output Shape: (5, num_classes)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.cnn1 = CNNBlock(in_channels, 50)
        self.cnn2 = CNNBlock(50, 100)

        self.fc1 = nn.Linear(100 * 4 * 4, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = ChristianModel(3, 7)

    x = torch.randn(3, 3, 16, 16)
    y = model(x)

    print(y)
