import csv
from typing import Optional

import torch
from accelerate import Accelerator
from transformers import EsmModel, EsmTokenizer

from enzrxnpred2.extension.seed import set_random_seed


def compute_sentence_embedding(data_path, model_path, output_csv, seed: int, max_seq_len: Optional[int],
                               pooling_type: str, mixed_precision: str, batch_size: int):
    set_random_seed(seed)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    with open(data_path) as f:
        sequences = [line for line in f.read().splitlines() if line.strip()]

    if max_seq_len is not None:
        lens = [len(seq) for seq in sequences]
        assert max(lens) <= max_seq_len, max(lens)
    model = EsmModel.from_pretrained(model_path)
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = accelerator.prepare(model)

    model.eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    all_pooled_outputs = []

    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i: i + batch_size]
        inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        if pooling_type == 'pooler':
            all_pooled_outputs.append(outputs.pooler_output.cpu())
        elif pooling_type == 'mean':
            hidden_states = outputs.last_hidden_state[:, 1:-1, :]
            embedding = hidden_states.mean(dim=1).cpu()
            all_pooled_outputs.append(embedding)
        else:
            raise ValueError('unexpected pooling type')
    pooled_outputs = torch.cat(all_pooled_outputs, dim=0)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "embedding"])
        for seq, vec in zip(sequences, pooled_outputs):
            writer.writerow([seq, vec.tolist()])
    print(f"Saved: '{output_csv}'")
