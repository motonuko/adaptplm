#!/bin/bash

# List up all the target keys
keys=(
  duf
  esterase
  gt_acceptors_chiral
  halogenase_NaBr
  olea
  phosphatase_chiral
)

mkdir -p build/blast
db="build/blast/enzsrp_full_db"
makeblastdb -in build/fasta/enzsrp_full/enzsrp_full_input.fasta -dbtype prot -out "${db}"

# Run BLAST for each target key
for key in "${keys[@]}"; do
    input_file="build/fasta/${key}/${key}_input.fasta"
    output_file="build/blast/${key}_enzsrp_full_results.tsv"

    if [[ -f "$input_file" ]]; then
        echo "🌀 Running blastp on $key "
        blastp -query "${input_file}" \
               -db "${db}" \
               -out "${output_file}" \
               -outfmt 6 -evalue 1e-4 -max_target_seqs 9999
    else
        echo "⚠️  Skipping: $input_file does not exist"
    fi
done
