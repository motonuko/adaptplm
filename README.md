## About

This repository contains main pipelines for model training and data processing used in our study.
Please cite the paper below when using this repository:


> Tomoya Okuno, Naoaki Ono, Md. Altaf-Ul-Amin, Shigehiko Kanaya, Self-supervised domain adaptation of protein language model based solely on positive enzyme-reaction pairs, Computational and Structural Biotechnology Journal, 27, 2025, 5441-5449, ISSN 2001-0370, https://doi.org/10.1016/j.csbj.2025.11.045.


## Environment Setup

Datasets for training were preprocessed on a laptop using a Conda environment.
Model domain adaptation training and downstream tasks were executed on the server using Singularity.

### Preprocessing Environment

Create the preprocessing environment from the provided YAML file:

```shell
conda env create -f environment.yml
```

### Training Environment

#### 1. Build the `adaptplm` package

```shell
pip install build
python -m build
```


#### 2. Set up the training environment

Use the provided `.def` file along with `requirements.txt` to replicate the environment:

```shell
singularity build --fakeroot adaptplm_v1_0_0.sif adaptplm_v1_0_0.def 
```

> [!TIP]
> If you are not familiar with Singularity, consider using Docker instead.  
> We do not provide a Dockerfile, but the `.def` file is based on a Docker image, and its format is quite similar to a
> Dockerfile.

---

## Reproducing the Experiment

Before running the experiment, create a directory to store the logs:

```shell
mkdir logs
```

Please refer to the appropriate document below according to your task.

- [README_MASKED_LM.md](docs/README_MASKED_LM.md)
    - For domain adaptation of ESM model with EnzSRP dataset.
- [README_DOWNSTREAM.md](docs/README_DOWNSTREAM.md)
    - For conducting downstream task experiments.
