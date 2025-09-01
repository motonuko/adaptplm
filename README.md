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

### Set up scripts

We use a python file from https://github.com/samgoldman97/enz-pred/ .
Please set up the python script by running the following script.

```shell
bin/set_up_parse_utils.sh
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

