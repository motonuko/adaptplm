## Environment setup

### Environment for preprocessing

Datasets for training are preprocessed on laptop using conda environment.

```shell
conda env create -f environment.yml
```

### Environment for training

The training was run on the server using Singularity (Both of domain adaptation and activity prediction).
Use the `.def` file and `requirements.txt` to replicate the environment.

```shell
singularity build --fakeroot enzrxnpred_v3.sif enzrxnpred_v3.def 
```

(If you are not familiar with Singularity, I recommend using Docker. We do not provide a Dockerfile,
but the `.def` file is based on a Docker image, and its format is quite similar to a Dockerfile.)


---

## Reproduction of the experiment

```shell
mkdir logs
```

- [README_MASKED_LM.md](README_MASKED_LM.md)
    - For domain adaptation of ESM model with EnzSRP dataset.
- [README_DOWNSTREAM.md](README_DOWNSTREAM.md)
    - For conducting downstream task experiments.

---

## References

CD-HIT word size reference
https://github.com/weizhongli/cdhit/blob/master/doc/cdhit-user-guide.wiki#algorithm-limitations
