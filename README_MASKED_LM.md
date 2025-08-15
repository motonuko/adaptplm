## 1. Prepare datasets

### EnzSRP (Enzyme Sequence Reaction Pair) dataset

(TODO: link to another project)


## 2. Clean and split dataset.

### Clean

```shell
enzrxn-preprocess clean-enzyme-reaction-pair-full-dataset > logs/clean-enzyme-reaction-pair-full-dataset.log 2>&1
enzrxn-preprocess clean-enzyme-reaction-pair-dataset > logs/clean-enzyme-reaction-pair-dataset.log 2>&1
```

### Split

```shell
enzrxn-preprocess create-sequence-inputs-for-splitting
mkdir -p build/cdhit
```

```shell
cd-hit -i build/fasta/enzsrp_full/enzsrp_full_input.fasta -o build/cdhit/enzsrp_full/enzsrp_full_80 -c 0.8 -n 5 -T 4 -M 4000 -d 0
enzrxn-preprocess split-enzsrp-full-dataset > logs/split-enzsrp-full-dataset.log 2>&1
```

## 3. Training

### Build vocab file

```shell
enzrxn-mlm build-vocab-enzsrp-full > logs/build-vocab-enzsrp-full.log 2>&1
```


### Train Cross-Attention Model

```shell
enzrxn-mlm train-seq-rxn-encoder-with-mlm --n-training-steps 1 --batch-size 2 --seq-rxn-encoder-config-file data/exp_configs/sample2024Nov/seq_rxn_encoder.json --bert-pretrained local/exp/rxn_encoder_train/241204_152116 --esm-pretrained facebook/esm2_t6_8M_UR50D
```

To use `--weighted-sampling` option, CD-HIT result is needed, which contains training sequences.
