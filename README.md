![test-badge](https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam/actions/workflows/test.yml/badge.svg) ![format](https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam/actions/workflows/format.yml/badge.svg) ![sphinx](https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam/actions/workflows/sphinx.yml/badge.svg) ![build-image](https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam/actions/workflows/build-image.yml/badge.svg)

# Collaborative-Coding-Exam
Repository for final evaluation in the FYS-8805 Reproducible Research and Collaborative coding course

## **Table of Contents**  
1. [Project Description](#project-description)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Results](#results)  
5. [Citing](#citing)  

## Project Description
This project involves collaborative work on a digit classification task, where each participant works on distinct but interconnected components within a shared codebase. <br>
The main goal is to develop and train digit classification models collaboratively, with a focus on leveraging shared resources and learning efficient experimentation practices.
### Key Aspects of the Project:
- **Individual and Joint Tasks:** Each participant has separate tasks, such as implementing a digit classification dataset, a neural network model, and an evaluation metric. However, all models and datasets must be compatible, as we can only train and evaluate using partners' models and datasets.
- **Shared Environment:** Alongside working on our individual tasks, we collaborate on joint tasks like the main file, and training and evaluation loops. Additionally, we utilize a shared Weights and Biases environment for experiment management.
- **Documentation and Package Management:** To ensure proper documentation and ease of use, we set up Sphinx documentation and made the repository pip-installable
- **High-Performance Computing:** A key learning objective of this project is to gain experience with running experiments on high-performance computing (HPC) resources. To this end, we trained all models on a cluster

## Installation

Install from:

```sh
pip install git+https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam.git
```

or using [uv](https://docs.astral.sh/uv/):

```sh
uv add git+https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam.git
```

To verify:

```sh
python -c "import CollaborativeCoding"
```

## Usage

To train a classification model using this code, follow these steps:

### 1) Create a Directory for the reuslts
Before running the training script, ensure the results directory exists:

 `mkdir -p "<RESULTS_DIRECTORY>"`

### 2) Run the following command for training, evaluation and testing

 `python3 main.py --modelname "<MODEL_NAME>" --dataset "<DATASET_NAME>" --metric "<METRIC_1>" "<METRIC_2>" ... "<METRIC_N>" --resultfolder "<RESULTS_DIRECTORY>" --run_name "<RUN_NAME>" --device "<DEVICE>"`
<br> Replace placeholders with your desired values:

- `<MODEL_NAME>`: You can choose from different models ( `"MagnusModel", "ChristianModel", "SolveigModel", "JanModel", "JohanModel"`).


- `<DATASET_NAME>`: The following datasets are supported (`"svhn", "usps_0-6", "usps_7-9", "mnist_0-3", "mnist_4-9"`)


- `<METRIC_1> ... <METRIC_N>`: Specify one or more evaluation metrics (`"entropy", "f1", "recall", "precision", "accuracy"`)


- `<RESULTS_DIRECTORY>`: Folder where all model outputs, logs, and checkpoints are saved 


- `<RUN_NAME>`: Name for WANDB project


- `<DEVICE>`: `"cuda", "cpu", "mps"`


## Running on a k8s cluster

In your job manifest, include:

```yaml
# ...
spec:
  template:
    spec:
      containers:
        - name: some-container-name
          image: "ghcr.io/sfi-visual-intelligence/collaborative-coding-exam:main"
          # ...
```

to pull the latest build, or check the [packages](https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam/pkgs/container/collaborative-coding-exam), for earlier versions.

> [!NOTE]
> The container is build for a `linux/amd64` architecture to properly build Cuda 12. For other architectures please build the docker image locally.


## Results 
### JanModel & MNIST_0-3
This section reports the results from using the model "JanModel" and the dataset MNIST_0-3 which contains MNIST digits from 0 to 3 (Four classes total). 
For this experiment we use all five available metrics, and train for a total of 20 epochs.

We achieve a great fit on the data. Below are the results for the described run:

| Dataset Split | Loss  | Entropy | Accuracy | Precision | Recall | F1    |
|---------------|-------|---------|----------|-----------|--------|-------|
| Train         | 0.000 | 0.000   | 1.000    | 1.000     | 1.000  | 1.000 |
| Validation    | 0.035 | 0.006   | 0.991    | 0.991     | 0.991  | 0.991 |
| Test          | 0.024 | 0.004   | 0.994    | 0.994     | 0.994  | 0.994 |


### MagnusModel & SVHN 
The MagnusModel was trained on the SVHN dataset, utilizing all five metrics.   
Employing micro-averaging for the calculation of F1 score, accuracy, recall, and precision, the model was fine-tuned over 20 epochs.   
A learning rate of 0.001 and a batch size of 64 were selected to optimize the training process. 

The table below presents the detailed results, showcasing the model's performance across these metrics.


| Dataset Split | Loss  | Entropy | Accuracy | Precision | Recall | F1    |
|---------------|-------|---------|----------|-----------|--------|-------|
| Train         | 1.007 | 0.998   | 0.686    | 0.686     | 0.686  | 0.686 |
| Validation    | 1.019 | 0.995   | 0.680    | 0.680     | 0.680  | 0.680 |
| Test          | 1.196 | 0.985   | 0.634    | 0.634     | 0.634  | 0.634 |

### ChristianModel & USPS_0-6
ChristianModel was trained on the USPS_0-6 dataset. The model was trained for a total of 20 epochs, utilizing all five metrics macro-averaged. Please find the results in the following table:

| Dataset Split | Loss  | Entropy | Accuracy | Precision | Recall | F1    |
|---------------|-------|---------|----------|-----------|--------|-------|
| Train         | 0.040 | 0.070   | 0.980    | 0.981     | 0.140  | 0.981 |
| Validation    | 0.071 | 0.074   | 0.973    | 0.975     | 0.140  | 0.974 |
| Test          | 0.247 | 0.096   | 0.931    | 0.934     | 0.134  | 0.932 |


### SolveigModel & USPS_7-9
SolveigModel was trained on the `USPS_7-9` dataset. The model was trained over 40 epochs, evaluating on all metrics with macro-averaging enabled, as shown in the table below.

| Dataset Split | Loss  | Entropy | Accuracy | Precision | Recall | F1    |
|---------------|-------|---------|----------|-----------|--------|-------|
| Train         | 0.013 | 0.017   | 1        | 1         | 0.333  | 1     |
| Validation    | 0.004 | 0.010   | 0.996    | 0.996     | 0.332  | 0.996 |
| Test          | 0.222 | 0.023   | 0.962    | 0.964     | 0.324  | 0.963 |

### JohanModel & MNIST_4-9
This section reports the results from using the model "JohanModel" and the dataset MNIST_4-9 which contains MNIST digits from 4 to 9 (six classes in total). 
All five available metrics were calculated for this experiment, model was trained for 12 epochs with learning rate of 0.001 and batch size 64.

The performance of the model is somewhat limited, at least compared with the results of JanModel on the other portion of MNIST dataset. Resulting metrics computed with macro-averaging below:

| Dataset Split | Loss  | Entropy | Accuracy | Precision | Recall | F1    |
|---------------|-------|---------|----------|-----------|--------|-------|
| Train         | 0.617 | 0.618   | 0.825    | 0.889     | 0.825  | 0.769 |
| Validation    | 0.677 | 0.619   | 0.809    | 0.872     | 0.809  | 0.755 |
| Test          | 0.679 | 0.618   | 0.810    | 0.870     | 0.810  | 0.755 |



## Citing
Please consider citing this repository if you end up using it for your work. 
Several citation methods can be found under the "About" section. 
For BibTeX citation please use
```
@software{Thrun_Collaborative_Coding_Exam_2025,
author = {Thrun, Solveig and Salomonsen, Christian and Størdal, Magnus and Zavadil, Jan and Mylius-Kroken, Johan},
month = feb,
title = {{Collaborative Coding Exam}},
url = {https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam},
version = {1.1.0},
year = {2025}
}
```

For APA please use
```
Thrun, S., Salomonsen, C., Størdal, M., Zavadil, J., & Mylius-Kroken, J. (2025). Collaborative Coding Exam (Version 1.1.0) [Computer software]. https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam
```
