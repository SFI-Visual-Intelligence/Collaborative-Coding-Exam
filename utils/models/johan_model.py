import torch.nn as nn

"""
Multi-layer perceptron model for image classification.
"""

# class NeuronLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()

#         self.fc = nn.Linear(in_features, out_features)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.fc(x)
#         x = self.relu(x)
#         return x


class JohanModel(nn.Module):
    """Small MLP model for image classification.

    Parameters
    ----------
    in_features : int
        Numer of input features.
    num_classes : int
        Number of classes in the dataset.

    """

    def __init__(self, in_features, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 77)
        self.fc2 = nn.Linear(77, 77)
        self.fc3 = nn.Linear(77, 77)
        self.fc4 = nn.Linear(77, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            x = layer(x)
            x = self.relu(x)
        x = self.softmax(x)
        return x


# TODO
# Add your tests here


if __name__ == "__main__":
    pass  # Add your tests here
