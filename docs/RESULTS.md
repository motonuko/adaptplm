
## Results

### Attention Analysis

```shell
enzrxn-downstream compute-clustering-scores \
  --seq-ec-file-path local/adaptplm_data/dataset/processed/embedding/embedding_evaluation_seq_ec.csv \
  --embedding-files local/adaptplm_data/outputs/embedding/embeddings_for_embeddings_evaluation_esm1b_t33_650M_UR50S.npz \
  --embedding-files local/adaptplm_data/outputs/embedding/embeddings_for_embeddings_evaluation_250420_121652.npz
```


### Activity classification for dense screen dataset

```shell
python adaptplm/viz/cpi_result/summarize_for_box_plot2.py --models 250420_121652 --result-path local/adaptplm_data/outputs/cpi
python adaptplm/viz/cpi_result/box_plot_3_x_2_v2.py > build/box_plot_3_x_2_v2.log 
```

### Binding site

Before running the code, download the RXNAAMAPPER results from https://doi.org/10.5281/zenodo.7530180 .

After downloading, add the following path to your .env file.

```
DATA_ORIGINAL_RXNAAMAPPER="<path-to-the-dir>/Biocatalysed_reaction_dataset"
```

```shell
python adaptplm/downstream/bindingsite/score_binding_site_pred.py --attn-indices-dir local/adaptplm_data/outputs/esm_attention/esm_250420_121652 > build/score_binding_site_pred.log
```