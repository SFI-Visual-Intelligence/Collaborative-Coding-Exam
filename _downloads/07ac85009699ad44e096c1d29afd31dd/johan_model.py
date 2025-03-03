import torch.nn as nn

"""
Multi-layer perceptron model for image classification.
"""


class JohanModel(nn.Module):
    """Small MLP model for image classification.

    Parameters
    ----------
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
    Grayscale images (like MNIST) have C = 1.
    Input shape: (N, 1, 28, 28)
    fc1 Output shape: (N, 77)
    fc2 Output shape: (N, 77)
    fc3 Output shape: (N, 77)
    fc4 Output shape: (N, num_classes)
    """

    def __init__(self, image_shape, num_classes):
        super().__init__()

        # Extract features from image shape
        self.in_channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.num_classes = num_classes
        self.in_features = self.in_channels * self.height * self.width

        self.fc1 = nn.Linear(self.in_features, 77)
        self.fc2 = nn.Linear(77, 77)
        self.fc3 = nn.Linear(77, 77)
        self.fc4 = nn.Linear(77, num_classes)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            x = layer(x)
            x = self.relu(x)
        return x


if __name__ == "__main__":
    print("This is JohanModel")
