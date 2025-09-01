## Downstream tasks

### 1. Prepare datasets for CD-HIT

To avoid running CD-HIT before each experiment, we provide a script that executes CD-HIT once for all downstream tasks.
Before running CD-HIT, please prepare the datasets for these tasks.

#### Collect the Training Dataset for ESP model

The fine-tuning training data of ESP model can be downloaded from the following link.

https://github.com/AlexanderKroll/ESP/blob/main/data/ESM1b_training/train_data_ESM_training.pkl

After downloading, please define the path to the downloaded file.

```shell
ORIGINAL_ESP_FINE_TUNING_PKL='<path-to-the-dir>/train_data_ESM_training.pkl'
```

### Collect Activity Screening Dataset

```shell
git clone git@github.com:samgoldman97/enzyme-datasets.git
```

```shell
DATA_ORIGINAL_DENSE_SCREEN_PROCESSED="<path-to-the-root-of-cloned-project>/data/processed"
```

### Collect K_cat Prediction Dataset

Download the data from https://zenodo.org/records/8367052

```shell
DATA_ORIGINAL_KCAT_PREDICTION_DATA="<path-to-the-downloaded-dir>/data"
ORIGINAL_KCAT_TRAIN_PKL="<path-to-the-downloaded-dir>/data/kcat_data/splits/train_df_kcat.pkl"
ORIGINAL_KCAT_TEST_PKL="<path-to-the-downloaded-dir>/data/kcat_data/splits/test_df_kcat.pkl"
```

### 2. Run CD-HIT

```shell
enzrxn-preprocess create-sequence-inputs-for-analysis
```

```shell
./bin/run_cdhit_60.sh
```

The following instructions are for each downstream task. Please refer to the section you’d like to run.

### 3. Embedding Evaluation with Clustering Score

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

```shell
enzrxn-downstream compute-clustering-scores \
  --seq-ec-file-path local/adaptplm_data/dataset/processed/embedding/embedding_evaluation_seq_ec.csv \
  --embedding-files local/adaptplm_data/outputs/embedding/embeddings_for_embeddings_evaluation_esm1b_t33_650M_UR50S.npz \
  --embedding-files local/adaptplm_data/outputs/embedding/embeddings_for_embeddings_evaluation_250420_121652.npz
```

### 4. Activity Classification for Family-wide Enzyme-substrate Specificity Screening Datasets

Create filtered enzyme-substrate Specificity Screening Datasets

```shell
python adaptplm/preprocess/cdhit2/cpi.py
```


```shell
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-downstream run-nested-cv-on-enz-activity-cls --exp-config "<path-to-expconfig>"
```

```shell
python adaptplm/viz/cpi_result/summarize_for_box_plot2.py --models 250420_121652 --result-path local/adaptplm_data/outputs/cpi
python adaptplm/viz/cpi_result/box_plot_3_x_2_v2.py > build/box_plot_3_x_2_v2.log 
```

### 5. k_cat Prediction

Create sequence list for embedding generation.

```shell
python adaptplm/data/extract_seq_kcat.py
```

Generate embeddings for enzymes.

```shell
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-downstream compute-sentence-embedding --data-path "data/dataset/processed/kcat/kcat_sequences.txt" --model-path "<path-to-the-adapted-esm-model>/esm" --output-csv "build/kcat/kcat_sequence_embeddings_<model-name>.csv" --batch-size 16
```

Generate a filtered list of kcat test sequence IDs that includes only sequences not similar to either the ESP fine-tuning training set or the EnzSRP training set.

```shell
python adaptplm/preprocess/cdhit2/kcat.py
```

The execution code for Kcat prediction is in a separate project. Please refer to https://github.com/motonuko/kcat_prediction_slim

### 6. Binding site prediction

```shell
singularity exec --nv adaptplm_v1_0_0.sif \
enzrxn-downstream extract-attention-order-per-head --data-path "build/rxnaamapper_sequences_1024.txt" --model-path "<path-to-the-adapted-esm-model>/esm" --output-dir "build/esm_attention/esm_attention_<model-name>"
```

```shell
python enzrxnpred2/downstream/bindingsite/score_binding_site_pred.py 250420_121652
```

Before running the next code, download the RXNAAMAPPER results from https://doi.org/10.5281/zenodo.7530180 .

After downloading, add the following path to your .env file.

```
DATA_ORIGINAL_RXNAAMAPPER="<path-to-the-dir>/Biocatalysed_reaction_dataset"
```

```shell
python adaptplm/downstream/bindingsite/score_binding_site_pred.py --attn-indices-dir local/adaptplm_data/outputs/esm_attention/esm_250420_121652 > build/score_binding_site_pred.log
```
