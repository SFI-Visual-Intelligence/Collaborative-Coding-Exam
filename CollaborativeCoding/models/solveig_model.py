import torch
import torch.nn as nn


def find_fc_input_shape(image_shape, model):
    """
    Finds the shape of the input to the fully connected layer after passing through the convolutional layers.

    This function takes an input image shape and the model's convolutional layers and computes
    the number of features passed into the first fully connected layer after the image has been processed
    through the convolutional layers.

    Code inspired by @Seilmast (https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam/issues/67#issuecomment-2651212254).

    Args
    ----
    image_shape : tuple(int, int, int)
        Shape of the input image (C, H, W), where C is the number of channels,
        H is the height, and W is the width of the image. This shape defines the input image dimensions.

    model : nn.Module
        The CNN model containing the convolutional layers. This model is used to pass the image through its
        layers to determine the output size, which is used to calculate the number of input features for the
        fully connected layer.

    Returns
    -------
    int
        The number of elements in the input to the fully connected layer after the image has passed
        through the convolutional layers. This value is used to initialize the size of the fully connected layer.
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
       A Convolutional Neural Network (CNN) model for classification.

       This model is designed for image classification tasks. It contains three convolutional blocks followed by
       a fully connected layer to make class predictions.

       Args
       ----
       image_shape : tuple(int, int, int)
           Shape of the input image (C, H, W), where C is the number of channels,
           H is the height, and W is the width of the image. This parameter defines the input shape of the image
           that will be passed through the network.

       num_classes : int
           The number of output classes for classification. This defines the size of the output layer (i.e., the
           number of units in the final fully connected layer).

       Attributes
       ----------
       conv_block1 : nn.Sequential
           The first convolutional block consisting of a convolutional layer, ReLU activation, and max-pooling.

       conv_block2 : nn.Sequential
           The second convolutional block consisting of a convolutional layer and ReLU activation.

       conv_block3 : nn.Sequential
           The third convolutional block consisting of a convolutional layer and ReLU activation.

       fc1 : nn.Linear
           The fully connected layer that takes the output from the convolutional blocks and outputs the final
           classification logits (raw scores for each class).

       Methods
       -------
       forward(x)
           Defines the forward pass of the network, which passes the input through the convolutional layers
           followed by the fully connected layer to produce class logits.
       """

    def __init__(self, image_shape, num_classes):
        """
        Initializes the SolveigModel with convolutional and fully connected layers.

        The model is constructed using three convolutional blocks, followed by a fully connected layer.
        The size of the input to the fully connected layer is determined dynamically based on the input image shape.

        Args
        ----
        image_shape : tuple(int, int, int)
            The shape of the input image (C, H, W) where C is the number of channels,
            H is the height, and W is the width.

        num_classes : int
            The number of classes for classification. This defines the output size of the final fully connected layer.
        """
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
        Defines the forward pass of the network.

        Args
        ----
        x : torch.Tensor
          A 4D tensor with shape (Batch Size, Channels, Height, Width) representing the input images.

        Returns
        -------
        torch.Tensor
          A 2D tensor of shape (Batch Size, num_classes) containing the logits (raw class scores)
          for each input image in the batch. These logits can be passed through a softmax function
          for probability values.
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        return x

