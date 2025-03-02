Magnus Individual Task
======================

## Task overview
In addition to the overall task, I was tasked to implement a three layer linear network, a dataset loader for the SVHN dataset, and a entropy metric.

## Network Implementation In-Depth
For the network part I was tasked with making a three-layer linear network where each layer conists of 133 neurons. This is a fairly straightforward implementation where we make a custom class which inherits from the PyTorch Module class. This allows for our class to have two methods. The __init__ method and a forward method. When we make an instance of the class we'll be able to call the instance like we would call a function, and have it run the forward method. 

The network is initialized with the following metrics: 
* image_shape
* num_classes
* nr_channels

The num_classes argument is used to define the number of output neurons. Each dataset has somewhere between 5 and 10 classes, and as such there isn't a single output size works well. 

As each layer is a linear layer we need to initialize the network with respect to the image size. We are working with datasets which are either greyscale or color images, and can be any height and width. Therefore we have the image_shape argument, which provides the information on the image height and width, and the nr_channels argument which states the number of channels we use. With these values we initialize the first layer accordingly, that is: height * width * channels inputsize. 

The forward method in this class has an assertion making sure the input has four channels, they being batch size, channels, height and width. 
Each input is flattened over the channel, height and width channels. Then they are passed through each layer and the resulting logits are returned.


## SVHN Dataset In-Depth
The dataloader I was tasked with making is to load the well-known SVHN dataset. This is a RGB dataset with real-life digits taken from house numbers. The class inherits from the torch Dataset class, and has four methods:
* __init__ : initialized the instance of the class
* _create_h5py: Creates the h5 object containing data from the downloaded .mat files for ease of use
* __len__ : Method needed in use of the DataLoader class. Returns length of the dataset
* __getitem__ : Method needed in use of the DataLoader class. Loads a image - label pair, applies any defined image transformations, and returns both image and label. 



The __init__ method takes in a few arguments. 
* data_path (Path): Path where either the data is downloaded or where it is to be downloaded to. 
* train (bool): Which set to use. If true we use the training set of SVHN, and if false we use the test set of SVHN.
* transform: The transform functions to be applied to the returned image. 
* nr_channels: How many channels to use. Can be either 1 or 3, corresponding to either greyscale or RGB images respectively. 

In the init we check for the existence of the SVHN dataset. If it does not exist, then we run the _create_h5py method which will be explained later. Then the labels are loaded into memory as they are needed for the __len__ method among other things. 

The _create_h5py method downloads a given SVHN set (train or test). We also change the label 10 to 0, as the SVHN dataset starts at index 1, with 10 representing images with the digit zero. After the download, we create two .h5 files. One with the labels and one with the images. 

Lastly, in __getitem__ we take index (number between 0 and length of label array). We retrive load the image h5 file, and retrive the row corresponding to the index. 
We then convert the image to an Pillow Image object, then apply the defined transforms before returning the image and label. 



## Entropy Metric In-Depth
The EntropyPrediction class' main job is to take some inputs from the MetricWrapper class and store the batchwise Shannon Entropy metric of those inputs. The class has four methods with the following jobs: 
* __init__ : Initialize the class.
* __call__ : Main method which is used to calculate and store the batchwise shannon entropy.
* __returnmetric__ : Returns the collected metric. 
* __reset__ : Removes all the stored values up until that point. Readies the instance for storing values from a new epoch. 

The __init__ method has two arguments, both present for compatability issues. However, the num_classes argument is used as a check in the __call__ method to assert the input is of correctly assumed size. 

In __call__ we get both true labels and model logit scores for each sample in the batch as input. We're calculating Shannon Entropy, not KL-divergence, so the true labels aren't actually needed. 
With permission I've used the scipy implementation to calculate entropy here. We apply a softmax over the logit values, then calculate the Shannon Entropy, and make sure to remove any Inf values which might arise from a perfect guess/distribution.



Next we have the __returnmetric__ method which is used to retrive the stored metric. This returns the mean over all stored values. Effectively, this will return the average Shannon Entropy of the dataset. 

Lastly we have the __reset__ method which simply emptied the variable which stores the entropy values to prepare it for the next epoch. 

## More on implementation choices
It should be noted that a lot of our decisions came from a top-down perspective. Many of our classes have design choices to accomendate the wrappers which handle the initialization and dataflow of the different metrics, dataloaders, and models. 
All in all, we've made sure you don't really need to interact with the code outside setting up the correct arguments for the run, which is great for consistency. 


## Challenges 
### Running someone elses code
This section answers the question on what I found easy / difficult running another persons code. 

I found it quite easy to run others code. We had quite good tests, and once every test passed, I only had one error with the F1 score not handeling an unexpected edgecase. To fix this I raised an issue, and it was fixed shortly after. 

One thing I did find a bit difficult was when people would change integral parts of the common code such as wrappers or loader functions (usually for the better), but did not raise an issue or notify about the change. It did cause some moments of questions, but in the end we sorted it out through weekly meetings where we agreed on design choices and how to handle loading of the different modules. 

The issues mentioned above also lead to a week or so where there was always a test failing, and the person whos' code was failing did not have time to work on it for a few days. 

### Someone running my code
This section answers the question on what I found easy / difficult having someone run my code. 

I did not experience that anyone had issues with my code. After I fixed all issues and tests related to my code, it seems to have run fine, and no issues have been raised to my awareness about this. 


## Tools
This section answers the question of which tools from the course I used during the home-exam. 

For this exam I used quite a few tools from the course. 
I've never used pytest and test functions while writing code. This was quite fun to learn how to use, and having github actions also run the same tests was a great addition. 

Github actions we used for quite a few things. We checked for code formatting, documentation generation and run the code tests. 

Using sphinx for documentation was also a great tool. Turns out it's possible to write the doc-string in such a way that it automatically generates the documentation for you. This has helped reduce the workload with documentation a lot, and makes writing proper docstrings worthwile. 