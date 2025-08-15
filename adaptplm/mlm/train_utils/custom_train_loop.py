import json
import os
import shutil
import sys
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm import tqdm
from transformers import get_scheduler

from adaptplm.extension.torch_ext import get_summary_writer


@dataclass(frozen=True)
class LoopConfig:
    n_training_steps: int
    eval_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    save_steps: int
    max_checkpoints: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    # https://github.com/facebookresearch/esm/issues/283
    mixed_precision: str
    randomize_rxn_smiles: bool

    def to_json_file(self, file_path: Path) -> None:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(asdict(self, dict_factory=self._dict_factory), file, ensure_ascii=False, indent=4)

    @staticmethod
    def _dict_factory(data):
        return {k: str(v) if isinstance(v, Path) else v for k, v in data}


class ModelCheckpointManager:
    def __init__(self, max_check_points: int):
        self.deque = deque(maxlen=max_check_points) if max_check_points > 0 else None

    def save_checkpoint(self, model: nn.Module, save_checkpoint_func, global_step: int):
        if self.deque is None:
            return
        if len(self.deque) == self.deque.maxlen:
            old_path = self.deque.popleft()
            if os.path.exists(old_path):
                if os.path.isfile(old_path):
                    os.remove(old_path)
                elif os.path.isdir(old_path):
                    shutil.rmtree(old_path)
                print(f'Old model checkpoint dir has been removed: {old_path}')
        save_path = save_checkpoint_func(model, global_step)
        self.deque.append(save_path)


def is_in_remainder_batches(batch_idx, loader_length, accumulation_steps):
    """
    Determines if the current batch index falls within the range of remainder batches.

    Parameters:
    - batch_idx (int): The current batch index (0-based).
    - loader_length (int): Total number of batches in the data loader.
    - accumulation_steps (int): Number of gradient accumulation steps.

    Returns:
    - bool: True if the batch falls within the remainder range, False otherwise.
    """
    remainder_batches_start = loader_length - (loader_length % accumulation_steps)
    return (batch_idx + 1) > remainder_batches_start


# NOTE: # NOTE: Warning: "Detected call of `lr_scheduler.step()` before `optimizer.step()`" may appear if
# `optimizer.step()` is skipped due to invalid gradients (e.g., NaN or Inf) and optimizer step is skipped. This behavior
# can depend on gradient accumulation settings and the device being used. If issues like gradient explosion or numerical
# instability (such as NaN or Inf) are observed, consider implementing gradient clipping. Ref:
# https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
# https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step-in-pytorch-1-1-0-and-later-you-should-call-them-in-the-opposite-order-optimizer-step-before-lr-scheduler-step/88295/4
#
# NOTE: Check pointing might be useful: https://huggingface.co/docs/accelerate/usage_guides/checkpoint#checkpointing
# NOTE: Tensorboard is used directory because it does work well with our custom implementation (Accelerator)
# Ref: https://huggingface.co/docs/accelerate/usage_guides/tracking
class CustomTrainLoop:
    def __init__(self, model, optimizer, train_data_loader, eval_data_loader, loop_config: LoopConfig, use_cpu,
                 out_dir):
        self.model = model
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.config = loop_config
        self.use_cpu = use_cpu
        self.out_dir = out_dir

    def train(self, save_model: Callable[[nn.Module], None], save_checkpoint: Callable[[nn.Module, int], None],
              label_key: str = 'labels'):
        writer = get_summary_writer(self.out_dir.as_posix(), is_debug=self.use_cpu)
        # pin_memory = device == "cuda"  # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723/2

        # could be moved to test code.
        model_params = {id(p): p for p in self.model.parameters()}
        optimizer_params = set(id(p) for group in self.optimizer.param_groups for p in group['params'])
        missing_params_ids = set(model_params.keys()) - optimizer_params
        if missing_params_ids:
            # missing_params = [model_params[param_id] for param_id in missing_params_ids]
            missing_params_info = [(name, p.shape) for name, p in self.model.named_parameters() if
                                   id(p) in missing_params_ids]
            raise ValueError(f"Missing parameters in optimizer: {missing_params_info}")

        scheduler = get_scheduler(name="linear", optimizer=self.optimizer,
                                  num_warmup_steps=0,  # default value of HF Trainer class
                                  num_training_steps=self.config.n_training_steps)
        accelerator = Accelerator(project_dir=self.out_dir, cpu=self.use_cpu,
                                  gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                                  mixed_precision=self.config.mixed_precision)
        model, optimizer, train_data_loader, scheduler = accelerator.prepare(self.model, self.optimizer,
                                                                             self.train_data_loader, scheduler)
        # Use accelerator in eval loop for mixed precision setting consistency.
        eval_data_loader = accelerator.prepare(self.eval_data_loader)

        checkpoint_manager = ModelCheckpointManager(self.config.max_checkpoints)

        progress_bar = tqdm(range(self.config.n_training_steps))

        global_step = 0
        train_cumulative_losses = []
        training_complete = False
        stopped_early = False
        best_eval_loss = float('inf')  # Initialize with infinity
        epochs_without_improvement = 0
        for epoch in range(1, sys.maxsize):
            model.train()
            cumulative_loss = 0
            for batch_idx, batch in enumerate(train_data_loader):
                eval_outputs = model(**batch)
                loss = eval_outputs.loss  # cross entropy per token mean across all masked tokens in batch
                # Scaling is needed since gradient_accumulation makes the total loss <gradient_accumulation_steps> times
                # bigger https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation
                accumulation_steps = accelerator.gradient_accumulation_steps
                if is_in_remainder_batches(batch_idx, len(train_data_loader), accumulation_steps):
                    # overwrite with remainder size. if remainder size == 0 returns original accumulation steps.
                    accumulation_steps = len(train_data_loader) % accumulation_steps or accumulation_steps
                loss = loss / accumulation_steps
                accelerator.backward(loss)
                cumulative_loss += loss.item()
                # NOTE: This is not efficient for multi-gpu environment. Use accelerator.accumulate(model) and
                # accelerator.sync_gradients
                if (batch_idx + 1) % accelerator.gradient_accumulation_steps == 0 or batch_idx + 1 == len(
                        train_data_loader):
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    progress_bar.update(1)
                    writer.add_scalar("PerUpdate/LearningRate", optimizer.param_groups[0]['lr'], global_step)
                    train_cumulative_losses.append(cumulative_loss)
                    writer.add_scalar('PerUpdate/CumulativeLoss', cumulative_loss, global_step)
                    optimizer.zero_grad()
                    cumulative_loss = 0
                    if global_step % self.config.save_steps == 0:
                        checkpoint_manager.save_checkpoint(model, save_checkpoint, global_step)
                    if global_step % self.config.eval_steps == 0:
                        train_batch_avg_loss = sum(train_cumulative_losses) / len(train_cumulative_losses)
                        writer.add_scalar('PerEvalStep/TrainingMeanCumulativeLoss', train_batch_avg_loss, global_step)
                        print(
                            f"Epoch {epoch}, Step {global_step}, Training Mean Cumulative Loss: {train_batch_avg_loss:.4f}")
                        train_cumulative_losses = []
                        model.eval()
                        total_loss = 0
                        total_tokens = 0
                        for eval_batch in eval_data_loader:
                            with torch.no_grad():
                                eval_outputs = model(**eval_batch)
                                eval_batch_loss = eval_outputs.loss
                                # not exact same as the mean of whole masked tokens in dataset since round error exists
                                num_tokens = eval_batch[label_key].ne(-100).sum().item()
                                total_loss += eval_batch_loss.item() * num_tokens
                                total_tokens += num_tokens
                        eval_avg_loss = total_loss / total_tokens
                        writer.add_scalar('PerEvalStep/ValMeanLoss', eval_avg_loss, global_step)
                        print(f"Val Mean Loss: {eval_avg_loss:.4f}")
                        if eval_avg_loss < best_eval_loss - self.config.early_stopping_min_delta:
                            best_eval_loss = eval_avg_loss
                            epochs_without_improvement = 0
                            save_model(self.model)
                        else:
                            epochs_without_improvement += 1
                        if epochs_without_improvement > self.config.early_stopping_patience:
                            print(f"Stopping early at epoch {epoch} step {global_step}.")
                            training_complete = True
                            stopped_early = True
                            break
                        model.train()
                if global_step >= self.config.n_training_steps:
                    training_complete = True
                    break
            if training_complete:
                break
        if not stopped_early:
            save_model(self.model)
        writer.close()
