# Individual task for Johan

## My implementation tasks 

* Data: Implement [MNIST](../CollaborativeCoding/dataloaders/mnist_4_9.py) dataset with digits between 4-9.
* Model: [MLP-model](../CollaborativeCoding/models/johan_model.py/) with 4 hidden layers, each with 77 neurons and ReLU activation.
* Evaluation metric: [Precision](../CollaborativeCoding/metrics/precision.py).

## Implementation choices

### Dataset 

The choices regarding the dataset were mostly done in conjunction with Jan (@hzavadil98) as we were both using the MNIST dataset. Jan had the idea to download the binary files and construct the images from those. The group decided collaboratorily to make the package download the data once and store it for all of use to use. Hence the individual implementations are fairly similar, at least for the two MNIST dataloaders. Were it not for these individual tasks, there would have been one dataloader class, initialised with two separate ranges for labels 0-3 and 4-9. However, individual dataloaders had to be created to comply with the exam description. For my implementation, the labels had to be mapped to a range starting at 0: $(4-9) \to (0,5)$ since the  cross-entropy loss function in PyTorch expect this range. 

## Experiences with running someone else's code 

## Experiences having someone else to run my code 

## I learned how to use these tools during this course 

### Git-stuff

### WandB

### Docker, Kubernetes and Springfield

### Proper documentation

### Nice ways of testing code 

### UV


