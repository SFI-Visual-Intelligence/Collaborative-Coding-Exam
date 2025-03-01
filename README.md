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

| Model     | Dataset   | Accuracy | Entropy | F1 Score | Precision | Recall |
|-----------|-----------|----------|---------|----------|-----------|--------|
| Christian |USPS (0-6) |          |         |          |           |        |
| Jan       |MNIST(0-3) |          |         |          |           |        |
| Johan     |MNITS (4-9)|          |         |          |           |        |
| Magnus    |SVHN       |          |         |          |           |        |
| Solveig   |USPS (7-9) |          |         |          |           |        |


## Citing
If you use this project in your research or work, please cite it as follows:

### **BibTeX Citation**  

```bibtex
@software{Collaborative_Coding_Exam_2025,
author = {Thrun, Solveig and Salomonsen, Christian and Størdal, Magnus and Zavadil, Jan and Mylius-Kroken, Johan},
month = feb,
title = {{Collaborative Coding Exam}},
url = {https://github.com/SFI-Visual-Intelligence/Collaborative-Coding-Exam},
version = {1.1.0},
year = {2025}
}
