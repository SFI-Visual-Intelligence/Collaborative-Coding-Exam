Magnus Individual Task
======================

# Magnus Størdal Individual Task

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




## Entropy Metric In-Depth

The EntropyPrediction class' main job is to take some inputs and return the Shannon Entropy metric of those inputs. The class has four methods with the following jobs: 
* __init__ : Initialize the class.
* __call__ : Main method which is used to calculate and store the batchwise shannon entropy.
* __returnmetric__ : Returns the collected metric. 
* __reset__ : Removes all the stored values up until that point. Readies the instance for storing values from a new epoch. 

The class is initialized with a single parameter called "averages". This is inspired from other PyTorch and NumPy implementations and controlls how values from different batches or within batches will be combined. The __init__ method checks the value of this argument with an assertion, which must be one of three string. We only allow "mean", "sum" and "none" as methods of combining the different entropy values. We'll come back to the specifics here.  
Furthermore, this method will also store the different Shannon Entropy values as we pass values into the __call__ method. 

In __call__ we get both true labels and model logit scores for each sample in the batch as input. We're calculating Shannon Entropy, not KL-divergence, so the true labels aren't needed. 
With permission I've used the scipy implementation to calculate entropy here. We apply a softmax over the logit values, then calculate the Shannon Entropy, and make sure to remove any NaN or Inf values which might arise from a perfect guess/distribution.

Next we have the __returnmetric__ method which is used to retrive the stored metric. Here the averages argument comes into play. 
Depending on what has been chosen as the averaging metric when initializing the class, one of the following operations will be applied to the stored values:
* Mean: Calculate the mean of the stored entropy values.
* Sum: Sum the stored entropy values.
* None: Do nothing with the stored entropy values. 
Then the value(s) are returned. 

Lastly we have the __reset__ method which simply emptied the variable which stores the entropy values to prepare it for the next epoch. 