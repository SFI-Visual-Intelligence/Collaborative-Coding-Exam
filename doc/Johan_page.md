# Individual task for Johan

## My implementation tasks

* Data: Implement [MNIST](../CollaborativeCoding/dataloaders/mnist_4_9.py) dataset with digits between 4-9.
* Model: [MLP-model](../CollaborativeCoding/models/johan_model.py) with 4 hidden layers, each with 77 neurons and ReLU activation.
* Evaluation metric: [Precision](../CollaborativeCoding/metrics/precision.py).

## Implementation choices

### Dataset

The choices regarding the dataset were mostly done in conjunction with Jan (@hzavadil98) as we were both using the MNIST dataset. Jan had the idea to download the binary files and construct the images from those. The group decided collaboratively to make the package download the data once and store it for all of us to use. Hence, the individual implementations are fairly similar, at least for the two MNIST dataloaders. Were it not for these individual tasks, there would have been one dataloader class, initialized with two separate ranges for labels 0-3 and 4-9. However, individual dataloaders had to be created to comply with the exam description. For my implementation, the labels had to be mapped to a range starting at 0: $(4-9) \to (0,5)$ since the cross-entropy loss function in PyTorch expect this range.

Dataset based on the PyTorch module ``Dataset``. 

* ``__init__`` Initialize the dataset. Shifts the labels $(4-9) \to (0,5)$ to comply with the expectations of the cross-entropy loss function.
  * ``data_path``: Path to where MNIST is/should be stored.
  * ``sample_ids``: Array of indices specifying which samples to load.
  * ``train``: Train or test the model (boolean), default is False.
  * ``transform``: Transforms if any, default is None.
  * ``nr_channels``: Number of channels, default is 1.
* ``__len__``: Return the length of dataloader.
* ``__getitem__``: Return the item of the specified index. Read the binary file at the correct position in order to generate the sample. Parameters:
  * ``idx``: Index of the desired sample. 

### Model

The model is a straightforward MLP that consists of 4 hidden layers, 77 neurons in each layer. The activation function is ReLU. The final output is logits.

The model inherits the basic class from PyTorch: ``nn.Module`` so that the only necessary methods are

* ``__init__``: Initialize the network with the following parameters:
  * ``ìmage_shape``: Shape of the image (channels, height, width).
  * ``num_classes``: Number of classes to predict. This controls the output dimension.
* ``forward``: One forward pass of the model. Parameters:
  * ``x`` One batch of data.

For any batch size ``N`` an example would be:

* Grayscale MNIST picture have shape (1,28,28).
* Input shape: (``N``,1,28,28)
* First layer output shape: (``N``,77)
* Second layer output shape: (``N``,77)
* Third layer output shape: (``N``,77)
* Fourth (final) layer output shape: (``N``, ``num_classes``)

### Metric

The precision metric is calculated as follows:

$$
Precision = \frac{TP}{TP+FP},
$$
where $TP$ and $FP$ are the numbers of true and false positives respectively. Hence, precision is a measure of how often the model is correct whenever it predicts the target class.

It can be calculated in two ways:

* Macro-averaging: The precision is calculated for each class separately and then averaged over (default).
* Micro-averaging: Find $TP$ and $FP$ for all the classes and calculate precision once with these values.

The precision metric is also subclass of the PyTorch ``nn.Module`` class. It has the following methods:

* ``__init__``: Class initialization. Creates the class variables ``y_true`` and ``y_pred`` which are used to calculate the metrics. Parameters are:
  * ``num_classes``: The number of classes.
  * ``macro_averaging``: Boolean flag that control how to average the precision. Default is false.
* ``forward``: Appends the true and predicted values to the class variables. Parameters:
  * ``y_true``: Tensor with true values.
  * ``y_pred``: Tensor with predicted values.
* ``_micro_avg_precision``: Computes the micro-averaged precision. Parameters:
  * ``y_true``: Tensor with true values.
  * ``y_pred``: Tensor with predicted values.
* ``_macro_avg_precision``: Computes the macro-averaged precision. Parameters:
  * ``y_true``: Tensor with true values.
  * ``y_pred``: Tensor with predicted values.
* ``__returnmetric__``: Return the micro/macro-averaged precision of the samples stored in the class variables ``y_true`` and ``y_pred``.
* ``__reset__``: Resets the list of samples stored in the class variables ``y_true`` and ``y_pred`` so that it is empty.

## Experiences with running someone else's code

This was an interesting experience as things did not go exactly as expected. I was initially unable to run with my assigned dataset (USPS 0-6). I have never encountered this error before, but with the help of Erik (IT guy) and Christian, a conflict with the ``urllib`` package and the newest python version was identifies. Christian recognized a solution, and suggested a fix which solved my problem. Hence, communicating well with the author of the code I was trying to run proved essential when I encountered errors.  

## Experiences having someone else to run my code

I have not heard anything from whoever ran my code, so I assume everything went well.

## I learned how to use these tools during this course

Coming from a non-machine-learning background where the coding routines are slightly different in the sense of code organization and the use of tools like "docker" and "WandB", the learning curve has been steep. I am grateful to have been given the chance to collaborate with skilled peers, from whom I have learned a lot.

### Git-stuff

My novel experience with git prior to this project consisted of mainly having a local repository where I pulled/pushed to a GitHub repo, using one branch only; version control for myself only. Here I learned to utilize the features of git when collaborating with others, which include:

* Operating with Issues and assignments of work division.
* Work on separate branches and have merge-protection on the main branch.
* Using pull requests where we had to review each other's code before merging into main.
* Use GitHub actions to automate workflows like unit testing and documentation building.
* Using tags and creating releases.
* Making the repository function like a package and make it installable.
* Writing clear and documented code and building documentations.

### WandB

It was insightful to learn the basics of using Weights and Biases to track the progress when training and evaluating models. I have used Tensorboard (slightly) prior to this, but WandB seems like a better option.

### Docker, Kubernetes and Springfield

Completely new to all of this. Spend quite some time trying to understand what docker and kubernetes is since we were supposed to run the experiment on the local cluster Springfield. There was a whole process setting everything up, making the ssh secrets and get everything to work. I have some prior experience with SSH protocols, but never used any container software, nor schedulers like kubernetes.

### Proper documentation

Writing good documentation is always necessary and having training in this was fruitful. Combining this with GitHub action and Sphinx made it far easier to have an updated version of the documentation readily available.

### Nice ways of testing code

The combination of having testing part of the GitHub action workflow and using the more advanced features of ``pytest`` (like parametrized testing) was new to be, a very nice thing to learn. It automated testing and made it significantly more easy to make sure that any code we pushed to the main branch was performing well, and did not lose any unintended functionality.

### UV

I switched to UV as my package manager for this project, and it is VERY good. Really fast and versatile.

