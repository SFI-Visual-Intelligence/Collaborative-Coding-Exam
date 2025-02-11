import torch.nn as nn


class MagnusModel(nn.Module):
    def __init__(self, image_shape: int, num_classes: int, imagechannels: int):
        """
        Magnus model contains the model for Magnus' part of the homeexam.
        This class contains a neural network consisting of three linear layers of 133 neurons each,
        with ReLU activation between each layer.

        Args
        ----
            image_shape (int): Expected size of input image. This is needed to scale first layer input
            imagechannels (int): Expected number of image channels. This is needed to scale first layer input
            num_classes (int): Number of classes we are to provide.

        Returns
        -------
            MagnusModel (nn.Module): Neural network as described above in this docstring.
        """
        super().__init__()
        self.image_shape = image_shape
        self.imagechannels = imagechannels

        self.layer1 = nn.Sequential(
            *(
                [
                    nn.Linear(
                        self.imagechannels * self.imagesize * self.imagesize, 133
                    ),
                    nn.ReLU(),
                ]
            )
        )
        self.layer2 = nn.Sequential(*([nn.Linear(133, 133), nn.ReLU()]))
        self.layer3 = nn.Sequential(*([nn.Linear(133, num_classes), nn.ReLU()]))

    def forward(self, x):
        """
        Forward pass of MagnusModel

        Args
        ----
            x (th.Tensor): Four-dimensional tensor in the form (Batch Size x Channels x Image Height x Image Width)

        Returns
        -------
            out (th.Tensor): Class-logits of network given input x
        """
        assert len(x.size) == 4

        x = x.view(x.size(0), -1)

        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)

        return out
