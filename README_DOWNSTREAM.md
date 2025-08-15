## Downstream tasks

### Collect Data from three different sources

#### The training dataset for ESP model

The fine-tuning training data of ESP model can be downloaded from the following link.

https://github.com/AlexanderKroll/ESP/blob/main/data/ESM1b_training/train_data_ESM_training.pkl

After downloading, please define the path to the downloaded file.

```shell
ORIGINAL_ESP_FINE_TUNING_PKL='<path-to-the-dir>/train_data_ESM_training.pkl'
```

### Activity screening dataset

```shell
git clone git@github.com:samgoldman97/enzyme-datasets.git
```

```shell
DATA_ORIGINAL_DENSE_SCREEN_PROCESSED="<path-to-the-root-of-cloned-project>/data/processed"
```

### Kcat prediction

Download the data from https://zenodo.org/records/8367052

```shell
DATA_ORIGINAL_KCAT_PREDICTION_DATA="<path-to-the-downloaded-dir>/data"
ORIGINAL_KCAT_TRAIN_PKL="<path-to-the-downloaded-dir>/data/kcat_data/splits/train_df_kcat.pkl"
ORIGINAL_KCAT_TEST_PKL="<path-to-the-downloaded-dir>/data/kcat_data/splits/test_df_kcat.pkl"
```

### Common process

```shell
enzrxn-preprocess create-sequence-inputs-for-analysis
```

```shell
./bin/run_cdhit_60.sh
```

### Embedding evaluation

```shell
python adaptplm/downstream/embedding/extract_sequences_for_embed_eval.py
```

Compute embedding of enzyme sequences in EnzSRP test set with using 1.mean, 2.our model.

```shell
# Mean pooling
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-downstream compute-sentence-embedding2 --data-path "data/dataset/processed/embedding/enzsrp_test_sequences.txt" --model-path "facebook/esm1b_t33_650M_UR50S" --output-npy "build/embed/embeddings_for_embeddings_evaluation_esm1b_t33_650M_UR50S.npz" --pooling-type mean --batch-size 16
```

```shell
# domain-adapted ESM-1b
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-downstream compute-sentence-embedding2 --data-path "data/dataset/processed/embedding/enzsrp_test_sequences.txt" --model-path "<path-to-the-adapted-esm-model>/esm" --output-npy "build/embed/embeddings_for_embeddings_evaluation_<model-name>.npz" --batch-size 16
```

### Train dense screen model


```shell
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-downstream run-nested-cv-on-enz-activity-cls --exp-config "<path-to-expconfig>"
```

### Train k_cat model

create sequence list for embedding generation

```shell
python adaptplm/data/extract_seq_kcat.py
```

```shell
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-downstream compute-sentence-embedding --data-path "data/dataset/processed/kcat/kcat_sequences.txt" --model-path "<path-to-the-adapted-esm-model>/esm" --output-csv "build/kcat/kcat_sequence_embeddings_<model-name>.csv" --batch-size 16
```

### Binding site prediction

```shell
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-downstream extract-attention-order-per-head --data-path "build/rxnaamapper_sequences_1024.txt" --model-path "<path-to-the-adapted-esm-model>/esm" --output-dir "build/esm_attention/esm_attention_<model-name>"
```
