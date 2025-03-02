# Jan Individual Task
======================

## Task Overview
In addition to the overall task, I was assigned the implementation of a multi-layer perceptron model, a dataset loader for a subset of the MNIST dataset, and an accuracy metric.

## Network Implementation In-Depth
For the network part, I was tasked with making a simple MLP network model for image classification tasks. The model consists of two hidden layers with 100 neurons each followed by a leaky-relu activation. This implementation involves creating a custom class that inherits from the PyTorch `nn.Module` class. This allows our class to have two methods: the `__init__` method and a `forward` method. When we create an instance of the class, we can call the instance like a function, which will run the `forward` method.

The network is initialized with the following parameters:
* `image_shape`
* `num_classes`

The `image_shape` argument provides the shape of the input image (channels, height, width) which is used to correctly initialize the input size of the first layer. The `num_classes` argument defines the number of output neurons, corresponding to the number of classes in the dataset. 

The forward method in this class processes the input as follows:
1. Flattens the input image.
2. Passes the flattened input through the first fully connected layer (`fc1`).
3. Applies a LeakyReLU activation function.
4. Passes the result through the second fully connected layer (`fc2`).
5. Applies another LeakyReLU activation function.
6. Passes the result through the output layer (`out`).

## MNIST Dataset In-Depth
For the dataset part, I was tasked with creating a custom dataset class for loading a subset of the MNIST dataset containing digits 0 to 3. This involved creating a class that inherits from the PyTorch `Dataset` class. 

The class is initialized with the following parameters:
* `data_path`
* `sample_ids`
* `train` (optional, default is False)
* `transform` (optional, default is None)
* `nr_channels` (optional, default is 1)

The `data_path` argument stores the path to the four binary files containing MNIST dataset. The verification of presence of these files and their download, if necessary, is facilitated by the `Downloader`class. The `sample_ids` parameter contains the indices of images and their respective labels that are to be loaded from MNIST dataset. Filtering and random splitting of these indices is performed within the `load_data`function. `train`is a boolean flag indicating whether to load data from training (for training and validation splits) or from testing (test split) part of the MNIST dataset. `transform` is a callable created with `torch.compose()` to be applied on the images. `nr_channels` is not used in this dataset, only included for compatibility with other functions.

The class has two main methods:
* `__len__`: Returns the number of samples in the dataset.
* `__getitem__`: Retrieves the image and label at the specified index.

## Accuracy Metric In-Depth
For the metric part, I was tasked with creating an accuracy metric class. The `Accuracy` class computes the accuracy of a model's predictions. The class is initialized with the following parameters:
* `num_classes`
* `macro_averaging` (optional, default is False)

The `num_classes` argument specifies the number of classes in the classification task. The `macro_averaging`argument is a boolean flag specifying whether to compute the accuracy using micro or macro averaging.

The class has the following methods:
* `forward`: Stores the true and predicted labels computed on a batch level.
* `_macro_acc`: Computes the macro-averaged accuracy on stored values.
* `_micro_acc`: Computes the micro-averaged accuracy on stored values.
* `__returnmetric__`: Returns the computed accuracy based on the averaging method for all stored predictions.
* `__reset__`: Resets the stored true and predicted labels.

The `forward` method takes the true labels and predicted labels as input and stores them. The `_macro_acc` method computes the macro-average accuracy by averaging the accuracy for each class. The `_micro_acc` method computes the micro-average accuracy by calculating the overall accuracy. The `__returnmetric__` method returns the computed accuracy based on the averaging method. The `__reset__` method resets the stored true and predicted labels to prepare for the next epoch.