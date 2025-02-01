import pytest
import torch
import torch.nn as nn


class SolveigModel(nn.Module):
    """
        A Convolutional Neural Network model for classification.

        Args:
        ----
        in_channels : int
            Number of input channels (e.g., 3 for RGB images, 1 for grayscale).
        num_classes : int
            The number of output classes (e.g., 2 for binary classification).

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

    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Define the first convolutional block (conv + relu + maxpool)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=25, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define the second convolutional block (conv + relu)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Define the third convolutional block (conv + relu)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(100 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = nn.Softmax(x)

        return x


if __name__ == "__main__":
    model = SolveigModel(3, 3)

    x = torch.randn(1, 3, 16, 16)
    y = model(x)

    print(y)
