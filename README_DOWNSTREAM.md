## Downstream tasks

### Common process

```shell
enzrxn-preprocess create-sequence-inputs-for-analysis
```

```shell
./bin/run_cdhit_60.sh
./bin/run_cdhit_40.sh
```

### Embedding evaluation

EnzSRP のテストセットに対して 1.mean, 2.ESP, 3.SeqRxnModel の埋め込みを生成し，評価する．

```shell
compute_sentence_embedding2
```

### Train dense screen model

Download data from https://github.com/samgoldman97/enzyme-datasets/



```shell
enzrxn-downstream run-nested-cv-on-enz-activity-cls --exp-config data/exp_configs/sample2024Nov/fnn_active_site_morgan_roc_auc_dist7_light.json
```

### Train k_cat model

```shell
python create_filtered_kcat_test.py
```

create sequence list for embedding generation

```shell
python extract_seq_kcat.py
```

```shell
compute-sentence-embedding
```

### Binding site prediction
