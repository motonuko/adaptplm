#!/bin/bash

# List up all the target keys
keys=(
#  enzsrp_full_train
  enzsrp_full_train__esp__kcat
#  duf
#  esterase
#  gt_acceptors_chiral
#  halogenase_NaBr
#  olea
#  phosphatase_chiral
)

# Run CD-HIT for each target key
for key in "${keys[@]}"; do
    input_file="build/fasta/${key}/${key}_input.fasta"
    output_file="build/cdhit/${key}/${key}_80"

    if [[ -f "$input_file" ]]; then
        mkdir -p "build/cdhit/${key}"
        echo "🌀 Running cd-hit on $key (80%)"
        cd-hit -i "$input_file" \
               -o "$output_file" \
               -c 0.8 -n 5 -T 4 -M 4000 -d 0
    else
        echo "⚠️  Skipping: $input_file does not exist"
    fi
done
