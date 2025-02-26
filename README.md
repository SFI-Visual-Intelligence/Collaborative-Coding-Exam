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

TODO: Fill in

### Running on a k8s cluster

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

| Model     | Accuracy | Entropy | F1 Score | Precision | Recall |
|-----------|----------|---------|----------|-----------|--------|
| Christian |          |         |          |           |        |
| Jan       |          |         |          |           |        |
| Johan     |          |         |          |           |        |
| Magnus    |          |         |          |           |        |
| Solveig   |          |         |          |           |        |


## Citing
