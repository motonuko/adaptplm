#!/bin/bash

# List up all the target keys
keys=(
  enzsrp_full_train__esp__kcat
#  enzsrp_full__esp
  enzsrp_full__duf
  enzsrp_full__esterase
  enzsrp_full__gt_acceptors_chiral
  enzsrp_full__halogenase_NaBr
  enzsrp_full__olea
  enzsrp_full__phosphatase_chiral
  enzsrp_full__turnup
)

# Run CD-HIT for each target key
for key in "${keys[@]}"; do
    input_file="build/fasta/${key}/${key}_input.fasta"
    output_file="build/cdhit/${key}/${key}_40"

    if [[ -f "$input_file" ]]; then
        mkdir -p "build/cdhit/${key}"
        echo "🌀 Running cd-hit on $key (60%)"
        cd-hit -i "$input_file" \
               -o "$output_file" \
               -c 0.4 -n 2 -T 4 -M 4000 -d 0
    else
        echo "⚠️  Skipping: $input_file does not exist"
    fi
done
