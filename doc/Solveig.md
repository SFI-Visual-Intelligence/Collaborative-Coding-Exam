 Solveig Individual Task
======================

#   Task overview
In addition to the overall task, I was assigned the implementation of the following tasks:
* A [CNN-model](../CollaborativeCoding/models/solveig_model.py)
* A dataset class for the [USPS dataset](../CollaborativeCoding/dataloaders/uspsh5_7_9.py) for the digits from 7-9 
* The [F1](../CollaborativeCoding/metrics/F1.py) evaluation metric

 

## Dataset: USPS
This class is a custom PyTorch Dataset that loads a subset of the USPS dataset, specifically images of the digits 7, 8, and 9, from an HDF5 file. 
It supports image transformations and provides methods to retrieve both images and their corresponding labels.
The original USPS dataset labels digits as 7, 8, 9, but PyTorch’s cross-entropy loss function (which is used for classification tasks) expects class labels starting from 0.
To address this, a label mapping function (`label_shift`)  is used to remap the labels to the range [0, 2], where 7 → 0, 8 → 1, and 9 → 2. 
Additionally, a reverse mapping function (`label_restore`) is defined to restore the original labels if needed.

* `USPSH5_Digit_7_9_Dataset` Inherits from: `torch.utils.data.Dataset`
    
  This means the class follows the PyTorch Dataset interface, which requires implementation of the `__len__` and `__getitem__` methods. This allows the dataset to be used with PyTorch's DataLoader for batch processing and iteration.

The class has three methods:
1)  `__init__(self, data_path, sample_ids, train=False, transform=None, nr_channels=1)`: This method initializes the USPS dataset by loading images and labels from the given .h5 file. The dataset is filtered to include only images of the digits 7, 8, and 9, and the labels are remapped to the range [0, 2]. It takes the following parameters:
     * `data_path`: The path to the directory containing the USPS .h5 file.
     * `sample_ids`:  A list of indices specifying which samples to load from the dataset.
     * `train`: If `True` the dataset will load the images from the training set, otherwise, it loads images from the test set. (optional, default is `False`)
     * `transform`: A transformation function applied to each image for data augmentation. (optional, default is `None`)
     * `nr_channels`:  The number of channels in the images. The USPS dataset consists of grayscale images, so the default is 1.

2)  `__len__(self)`: This method returns the number of images in the dataset.
3)  `__getitem__(self, id)`:  This method retrieves an image and its corresponding label from the dataset given an index `id`. If `transform`== `True`, the specified transformations are applied to the image.





## Model: CNN
For the model, I was tasked to implement a simple Convolutional Neural Network (CNN) which is named `SolveigModel`.  
The `SolveigModel` class inherits from `torch.nn.Module` and is designed for image classification tasks. 
The model is flexible and can handle input images of different shapes. It dynamically determines the input size to the fully connected layer based on the input image dimensions.
Additionally, the number of output classes is customizable.

In PyTorch, any model that inherits from `torch.nn.Module` must implement the `__init__` and `forward` methods. 
The `__init__` method is used to define the layers of the model, while the `forward` method defines the flow of data through these layers.

Besides these, the `SolveigModel` also includes the method `find_fc_input_shape`. 
Instead of manually calculating the number of input features for the FC layer, this helper function automatically computes the number of features passed to it after the image has been processed by the convolutional blocks. 
This dynamic approach ensures that the model can handle input images of varying sizes without requiring manual adjustments.

The network is initialized with the following parameters:
* `image_shape`: represents the input image shape (cannels, height, width)
* `num_classes`: number of classes for the classification, which corresponds to the number of neurons in the final fully connected layer 

### Network structure:
The assignment required a simple three-layer CNN, and I structured it with progressively increasing filters (25 → 50 → 100). 
This helps the network learn low-level (edges, textures) in early layers and high-level (shapes, objects) features in deeper layers.

 The architecture consists of three sequential blocks (`conv_block1`, `conv_block2`, `conv_block`) followed by a fully connected layer (`fc1`).
* `conv_block1`: 2D convolutional layer with a kernel size of 3x3 followed by a ReLu activation and max-pooling with a kernel size 2x2
* `conv_block2`: 2D convolutional layer with a kernel size of 3x3 followed by a ReLu activation, no max-pooling
* `conv_block3`: 2D convolutional layer with a kernel size of 3x3 followed by a ReLu activation, no max-pooling
* `fc1`: takes the flattened features from the convolutional blocks and predict the class logits

### Forward pass of an image through the network
The forward function accepts a tensor of shape `(batch_Size, channels, height, width)`, representing a batch of input images.
First, the input passes through  `conv_block1`, `conv_block2`, and `conv_block3` for feature extraction. 
The output from these convolutional blocks is then flattened into a 2D tensor of shape  `(batch_size, num_features)`, ensuring it can be passed to the fully connected layer  `fc1`.  
Finally, the fully connected layer outputs a tensor of shape  `(batch_size, num_classes`), which contains the class logits.


## Metric: F1 
The F1 score is a metric used to evaluate the performance of a classification model and is defined as 
$$
F1 = 2 \frac{Precision * Recall}{Precision+Recall} = 2 \frac{TP}{2TP + FP + FN}.
$$
The class `F1Score` inherits from torch.nn.Module, which is the base class for all neural network modules in PyTorch. 
This allows it to seamlessly integrate into the PyTorch framework for model evaluation during training and inference. 
By inheriting from `torch.nn.module`, you can use this metric in the same way you would use other PyTorch metrics or layers.
Since  `F1Score` inherits from `torch.nn.module`, it must implement the `__init__` and `forward` methods. 
These methods define the initialization and the computation process for the F1 score, respectively.

It supports both micro-averaged and macro-averaged F1 scores, making it suitable for different types of classification problems.
* **Micro-averaged F1 Score:** This method computes the F1 score globally by treating all predictions as equally important. It calculates the total true positives (TP), false positives (FP), and false negatives (FN) across all classes and then computes the F1 score. Micro averaging is often used when you have an imbalanced dataset and want to emphasize the performance across all instances equally, regardless of class.
* **Macro-averaged F1 Score:** This method calculates the F1 score for each class independently and then averages the F1 scores across all classes. This is useful when you want to evaluate the model's performance per class, which helps to understand how well the model performs across different classes, even if they are imbalanced.

The class is initalized with two parameters:
* `num_classes`: The number of classes in the classification task
* `macro_averaging`:  If `True` the macro-averaged F1 score is computed else the micro-averaged F1 score (default = `False`)

### Methods:
* `forward(target, preds)`: This method first converts the logits (`preds`) to class indices and then stores the true labels (`target`) and predicted class indices in a list for the F1 computation. Storing the true labels and predictions allows for computing the F1 score over all the predictions after the entire dataset has been processed, instead of computing it batch-wise.
* `_micro_F1(target, preds)`: This method computes and returns the micro-averaged F1 score
* `_macro_F1(target, preds)`: This method computes and returns the macro-averaged F1 score
* `__returnmetric__`: This method computes and returns the F1 score based on the stored predictions and targets. If `macro_averaging` is `True` it computes the macro_averaged F1 score using the method `_macro_F1` else it uses the method `_micro_F1`.
* `__reset__`: This method is essential to clear the stored predictions and true labels after each epoch. It resets the lists with the stored values, ensuring that they do not accumulate values across multiple epochs.



# Experiences running another person's code
At the beginning of our project, we adopted a common code structure that would accommodate various wrappers managing the initialization and data flow of metrics, dataloaders, and models. 
To ensure smooth collaboration, we leveraged GitHub Actions for automated testing, which helped us validate the functionality of our dataloaders, metrics, and models. 
Once the various classes passed their tests, running code from other team members was no problem.

However, in the beginning, I struggled to keep track of all the changes in the GitHub repository due to large pull requests containing multiple modifications, which made it hard to follow and understand the updates.
To address this, we decided in our meetings to create a separate issue for each change. 
Subsequent pull requests would then address individual issues rather than combining multiple changes into a single pull request. 
This approach made it much easier to understand the changes.



# Experiences another person running my code
During the process of another person running my code, I got informed about an issue with my F1 metric, where it failed to handle an unexpected edge case at runtime—something that the initial tests did not catch. 
Once the issue was raised, I was promptly informed, and I addressed the error. 
However, as mentioned earlier, there were times when large pull requests included multiple important changes, making it challenging to adjust my code accordingly. 
Despite this, the collaborative nature of the process allowed us to identify and resolve issues effectively, improving the overall stability and functionality of the code.




# Tools 
During the collaborative coding course, I had the opportunity to learn and work with several valuable tools that were new to me:
* **Docker**
* **Ruff and Isort**
* **Github Actions**
* **Sphinx documentation**

During these course I realized how essential it is to have testing functions, especially in collaborative projects. 
Testing, along with GitHub Actions for automated workflows, helped keep the code robust and error-free. 
The Sphinx documentation tool also proved invaluable for generating project docs and tracking classes, making the codebase easier to understand. 
I also gained a lot of new knowledge about using Git, managing issues, assigning tasks, creating tags and releases, and making a repo pip-installable. 
Additionally, I learned how to add citations and licenses, which are crucial for proper project management.
These tools significantly boosted my productivity and understanding of best practices in collaborative software development and code documentation.

