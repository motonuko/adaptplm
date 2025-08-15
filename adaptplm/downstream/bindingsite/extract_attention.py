import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel

from adaptplm.extension.seed import set_random_seed
from adaptplm.extension.torch_ext import get_device
from adaptplm.mlm.train_utils.custom_data_collator4 import CustomTokenizedTextDataset4, MyCustomDataCollator4


def extract_attention_order_per_head(data_path, model_path, output_dir, seed: int):
    set_random_seed(seed)
    model_name = model_path.as_posix().split('/')[-1]  # TODO: using stem may be better.
    print(model_name)
    out = output_dir / model_name
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = EsmTokenizer.from_pretrained(model_path)
    eval_dataset = CustomTokenizedTextDataset4(data_path, tokenizer)
    model = EsmModel.from_pretrained(model_path)
    collator = MyCustomDataCollator4(tokenizer=tokenizer)
    # pin_memory = device == "cuda"  # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2
    eval_data_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=collator,
                                  num_workers=0,
                                  pin_memory=False)
    device = get_device(use_cpu=False)
    model.to(device)
    # accelerator = Accelerator(project_dir=t_config.out_parent_dir, cpu=True)  NOTE: Accelerator can not be used because batch contains non-tensor items
    # model, eval_data_loader = accelerator.prepare(model, eval_data_loader)
    model.eval()
    for batch in tqdm(eval_data_loader):
        data_ids = batch.pop('data_ids')
        labels = batch.pop('labels')
        batch = {key: value.to(device) for key, value in batch.items()}
        # NOTE: all tensors in batch were already moved on the device
        outputs = model(**batch, output_attentions=True)
        last_layer_attention = outputs.attentions[
            -1]  # last layer (batch_size, num_heads, seq_length(q), seq_length(k))
        # Attention to [CLS] token (batch_size, num_heads, seq_length), Normalized in query direction, not normalized in key direction.
        cls_attention_last_layer = last_layer_attention[:, :, 1:-1, 0]  # batch_size, num_heads, seq_length
        for data_id, attn in zip(data_ids, cls_attention_last_layer):
            seq_data = {}
            for head_idx, head_attn in enumerate(attn):
                sorted_indices = torch.argsort(head_attn, descending=True).cpu().tolist()
                assert head_attn[sorted_indices[0]] >= head_attn[sorted_indices[-1]]
                seq_data[f"head_{head_idx + 1}"] = sorted_indices
            with open(out / f"attn_indices_descending_{data_id}.json", "w", encoding="utf-8") as f:
                json.dump(seq_data, f, ensure_ascii=False, indent=4)
            # torch.save(attn, output_dir / f"attn_{data_id}.pt")


