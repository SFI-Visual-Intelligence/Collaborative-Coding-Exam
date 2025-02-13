import torch
import torch.nn as nn


class SolveigModel(nn.Module):
    """
    A Convolutional Neural Network model for classification.

     Args
    ----
    image_shape : tuple(int, int, int)
        Shape of the input image (C, H, W).
    num_classes : int
        Number of classes in the dataset.

    Attributes:
    -----------
    conv_block1 : nn.Sequential
        First convolutional block containing a convolutional layer, ReLU activation, and max-pooling.
    conv_block2 : nn.Sequential
        Second convolutional block containing a convolutional layer and ReLU activation.
    conv_block3 : nn.Sequential
        Third convolutional block containing a convolutional layer and ReLU activation.
    fc1 : nn.Linear
        Fully connected layer that outputs the final classification scores.
    """

    def __init__(self, image_shape, num_classes):
        super().__init__()

        C, *_ = image_shape

        # Define the first convolutional block (conv + relu + maxpool)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=25, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Define the second convolutional block (conv + relu)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Define the third convolutional block (conv + relu)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(100 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 16, 16)

    model = SolveigModel(x.shape[1:], 3)

    y = model(x)

    print(y)
