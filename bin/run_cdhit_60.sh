#!/bin/bash

# List up all the target keys
keys=(
  enzsrp_full_train__esp__kcat

  enzsrp_full_train__esp__duf
  enzsrp_full_train__esp__esterase
  enzsrp_full_train__esp__gt_acceptors_chiral
  enzsrp_full_train__esp__halogenase_NaBr
  enzsrp_full_train__esp__olea
  enzsrp_full_train__esp__phosphatase_chiral
#  enzsrp_full_train__esp
#  esp__duf
#  esp__esterase
#  esp__gt_acceptors_chiral
#  esp__halogenase_NaBr
#  esp__olea
#  esp__phosphatase_chiral
#  esp__turnup
)

# Run CD-HIT for each target key
for key in "${keys[@]}"; do
    input_file="build/fasta/${key}/${key}_input.fasta"
    output_file="build/cdhit/${key}/${key}_60"

    if [[ -f "$input_file" ]]; then
        mkdir -p "build/cdhit/${key}"
        echo "🌀 Running cd-hit on $key (60%)"
        cd-hit -i "$input_file" \
               -o "$output_file" \
               -c 0.6 -n 4 -T 4 -M 4000 -d 0
    else
        echo "⚠️  Skipping: $input_file does not exist"
    fi
done
