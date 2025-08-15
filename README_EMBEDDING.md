`.env`が読み込めないのでルートで実行すること

[//]: # (```shell)

[//]: # (enzrxn-downstream compute-sentence-embedding2 --data-path tests/data/sequence.txt --model-path facebook/esm2_t6_8M_UR50D --output-npy tests/data/temp_seq_embeddings.npy)

[//]: # (```)

[//]: # ()

[//]: # (```shell)

[//]: # (enzrxn-downstream compute-sentence-embedding --data-path tests/data/sequence.txt --model-path facebook/esm2_t6_8M_UR50D --output-csv tests/data/temp_seq_embeddings.csv)

[//]: # (```)

```shell
enzrxn-preprocess create-cdhit-input-activity-screen > logs/create-cdhit-input-activity-screen.log 2>&1
```

```shell
mkdir build/cdhit_activity_screen_output
cd-hit -i build/cdhit_activity_screen_input/activity_screen_duf.fasta -o build/cdhit_activity_screen_output/activity_screen_duf -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_esterase.fasta -o build/cdhit_activity_screen_output/activity_screen_esterase -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_gt_acceptors_chiral.fasta -o build/cdhit_activity_screen_output/activity_screen_gt_acceptors_chiral -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_halogenase_NaBr.fasta -o build/cdhit_activity_screen_output/activity_screen_halogenase_NaBr -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_olea.fasta -o build/cdhit_activity_screen_output/activity_screen_olea -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_phosphatase_chiral.fasta -o build/cdhit_activity_screen_output/activity_screen_phosphatase_chiral -c 0.6 -n 4 -T 4 -M 4000 -d 0
```

```shell
mkdir build/cdhit_activity_screen_output
cd-hit -i build/cdhit_activity_screen_input/activity_screen_duf.fasta -o build/cdhit_activity_screen_output/activity_screen_duf -c 0.4 -n 2 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_esterase.fasta -o build/cdhit_activity_screen_output/activity_screen_esterase -c 0.4 -n 2 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_gt_acceptors_chiral.fasta -o build/cdhit_activity_screen_output/activity_screen_gt_acceptors_chiral -c 0.4 -n 2 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_halogenase_NaBr.fasta -o build/cdhit_activity_screen_output/activity_screen_halogenase_NaBr -c 0.4 -n 2 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_olea.fasta -o build/cdhit_activity_screen_output/activity_screen_olea -c 0.4 -n 2 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_activity_screen_input/activity_screen_phosphatase_chiral.fasta -o build/cdhit_activity_screen_output/activity_screen_phosphatase_chiral -c 0.4 -n 2 -T 4 -M 4000 -d 0
```

## for CPI similarity evaluation

```shell
enzrxn-preprocess create-cdhit-input-esp-activity-screen > logs/create-cdhit-input-esp-activity-screen.log 2>&1
```

```shell
mkdir build/cdhit_output_activity_screen_esp
cd-hit -i build/cdhit_input_activity_screen_esp/esp_duf.fasta -o build/cdhit_output_activity_screen_esp/esp_duf -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_input_activity_screen_esp/esp_esterase.fasta -o build/cdhit_output_activity_screen_esp/esp_esterase -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_input_activity_screen_esp/esp_gt_acceptors_chiral.fasta -o build/cdhit_output_activity_screen_esp/esp_gt_acceptors_chiral -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_input_activity_screen_esp/esp_halogenase_NaBr.fasta -o build/cdhit_output_activity_screen_esp/esp_halogenase_NaBr -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_input_activity_screen_esp/esp_olea.fasta -o build/cdhit_output_activity_screen_esp/esp_olea -c 0.6 -n 4 -T 4 -M 4000 -d 0
cd-hit -i build/cdhit_input_activity_screen_esp/esp_phosphatase_chiral.fasta -o build/cdhit_output_activity_screen_esp/esp_phosphatase_chiral -c 0.6 -n 4 -T 4 -M 4000 -d 0
```
