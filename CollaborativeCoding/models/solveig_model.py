import torch
import torch.nn as nn


def find_fc_input_shape(image_shape, model):
    """
    Find the shape of the input to the fully connected layer after passing through the convolutional layers.

    Code inspired by @Seilmast (https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam/issues/67#issuecomment-2651212254)

    Args
    ----
    image_shape : tuple(int, int, int)
        Shape of the input image (C, H, W), where C is the number of channels,
        H is the height, and W is the width of the image.
    model : nn.Module
        The CNN model containing the convolutional layers, whose output size is used to
        determine the number of input features for the fully connected layer.

    Returns
    -------
    int
        The number of elements in the input to the fully connected layer.
    """

    dummy_img = torch.randn(1, *image_shape)
    with torch.no_grad():
        x = model.conv_block1(dummy_img)
        x = model.conv_block2(x)
        x = model.conv_block3(x)
        x = torch.flatten(x, 1)

    return x.size(1)


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

        fc_input_size = find_fc_input_shape(image_shape, self)

        self.fc1 = nn.Linear(fc_input_size, num_classes)

    def forward(self, x):
        """
        Defines the forward pass.
        Args:
            x (torch.Tensor): A four-dimensional tensor with shape
                        (Batch Size, Channels, Image Height, Image Width).
        Returns:
        torch.Tensor: The output tensor containing class logits for each input sample.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 28, 28)

    model = SolveigModel(x.shape[1:], 3)

    y = model(x)

    print(y)
