## 1. Prepare Datasets

### EnzSRP (Enzyme Sequence Reaction Pair) Dataset

Download EnzSRP dataset from the link.

(TODO: link to the file)

Place the EnzSRP dataset `enzsrp_full.csv` at `<project root>/data/dataset/raw/enzsrp_full.csv`

> [!TIP]
> For instructions on creating the EnzSRP dataset, refer to the project below.
> https://github.com/motonuko/enzsrp

## 2. Clean and split Dataset.

### Set Up .env File

We manage all data paths using the .env file. 
These paths will serve as the default paths, simplifying the process of running scripts.
Please create a .env file in the project's root directory:

```shell
touch .env
```

Set the path of project root in created `.env` file.

```shell
ENZRXNPRED_PROJECT_PATH="<path-to-the-project-root>"
```

### Clean Up EnzSRP Dataset

Before splitting EnzSRP dataset, data cleaning is conducted.

```shell
enzrxn-preprocess clean-enzyme-reaction-pair-full-dataset > logs/clean-enzyme-reaction-pair-full-dataset.log 2>&1
```

### Run CD-HIT

Prepare a sequence list and a directory for running CD-HIT using the following script:

```shell
enzrxn-preprocess create-sequence-inputs-for-splitting
mkdir -p build/cdhit/enzsrp_full
```

[!TIP]
> If cd-hit has not installed in your computer, please install it before running the following script.
> (the following script is only for mac user)
> ```shell
> brew tap brewsci/bio
> brew install cd-hit
> ```

Run CD-HIT 

```shell
cd-hit -i build/fasta/enzsrp_full/enzsrp_full_input.fasta -o build/cdhit/enzsrp_full/enzsrp_full_80 -c 0.8 -n 5 -T 4 -M 4000 -d 0
```

### Split Data with Using CD-HIT Result

```shell
enzrxn-preprocess split-enzsrp-full-dataset > logs/split-enzsrp-full-dataset.log 2>&1
```

## 3. Training

### Build vocab file

Run the following command to build the vocab file for chemical reaction encoder:

```shell
enzrxn-mlm build-vocab-enzsrp-full > logs/build-vocab-enzsrp-full.log 2>&1
```


### Fine-tune ESM model in Two-encoders Model

Finally, you can run the model training script by the following script:

```shell
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-mlm train-seq-rxn-encoder-with-mlm --bert-model-config-file "data/exp_configs/maskedlm/bert_enzsrp_full_2hl.json" --esm-pretrained  facebook/esm1b_t33_650M_UR50S --seq-rxn-encoder-config-file "data/exp_configs/maskedlm/seq_rxn_encoder_2_4.json" --batch-size 4 --gradient-accumulation_steps 4 --train-data-path "data/dataset/processed/enzsrp_full_cleaned/enzsrp_full_cleaned_train.csv" --eval-data-path "data/dataset/processed/enzsrp_full_cleaned/enzsrp_full_cleaned_val.csv" --vocab-path "data/dataset/processed/vocab/enzsrp_full_cleand_train_vocab.txt" --randomize-rxn-smiles
```


## References

CD-HIT word size reference
https://github.com/weizhongli/cdhit/blob/master/doc/cdhit-user-guide.wiki#algorithm-limitations
