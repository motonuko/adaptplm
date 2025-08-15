import copy
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader


# for hyperparameter optimization
def train_and_eval(model, x1_train, x2_train, y_train, x1_val, x2_val, y_val, n_steps: int, eval_steps: int,
                   device: str, batch_size: int, learning_rate: float, weight_decay: float, scoring):
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train.view(-1).numpy().astype(int),
    )
    class_weights = torch.FloatTensor(class_weights)
    train_sample_weights = class_weights[y_train.long()]
    # NOTE: Use class weights from the training set for validation to ensure consistency.
    val_sample_weights = class_weights[y_val.long()]

    train_dataset = TensorDataset(x1_train, x2_train, y_train, train_sample_weights)
    val_dataset = TensorDataset(x1_val, x2_val, y_val, val_sample_weights)
    # NOTE: Instead of WeightedRandomSampler, we use class weights in loss function.
    # NOTE: Encounters error if set num_workers > 0 because using joblib outside.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True  # To avoid crash on BatchNorm if batch size == 1
                              )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_model_state = None
    best_score = float('inf')
    best_step = None
    score_history = []
    global_step = 0
    for _ in range(sys.maxsize):
        model.to(device)
        model.train()
        for batch in train_loader:
            batch_x1, batch_x2, batch_y, batch_weight = (t.to(device) for t in batch)
            criterion = nn.BCEWithLogitsLoss(batch_weight)
            optimizer.zero_grad()
            output = model(batch_x1, batch_x2)
            loss = criterion(output['logit'], batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_steps == 0:
                _, score = _eval_loop(model, val_loader, device, scoring)
                score_history.append(score)
                if score < best_score:
                    best_score = score
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_step = global_step
            if global_step >= n_steps:
                break
        if global_step >= n_steps:
            break

    model.load_state_dict(best_model_state)  # load best model
    _, best_model_score = _eval_loop(model, val_loader, device, scoring)
    # best_model_score = scoring(trues, preds)
    return best_model_score, model, best_step, score_history


# for final model training
def train_loop(model, x1_train, x2_train, y_train, n_steps: int, device: str,
               batch_size: int, learning_rate: float, weight_decay: float, log_dir: Path):
    # writer = get_summary_writer(log_dir.as_posix())

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train.view(-1).numpy().astype(int),
    )
    class_weights = torch.FloatTensor(class_weights)
    train_sample_weights = class_weights[y_train.long()]
    train_dataset = TensorDataset(x1_train, x2_train, y_train, train_sample_weights)
    # NOTE: Encounters error if set num_workers > 0 because using joblib.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True  # To avoid crash on BatchNorm if batch size == 1
                              )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    global_step = 0
    for _ in range(sys.maxsize):
        model.to(device)
        model.train()
        for batch in train_loader:
            batch_x1, batch_x2, batch_y, batch_weight = (t.to(device) for t in batch)
            criterion = nn.BCEWithLogitsLoss(batch_weight)
            optimizer.zero_grad()
            output = model(batch_x1, batch_x2)
            loss = criterion(output['logit'], batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()
            global_step += 1
            # writer.add_scalar('training_loss', loss.item(), global_step)
            if global_step >= n_steps:
                break
        if global_step >= n_steps:
            break
    # writer.close()
    return model


def _eval_loop(model: nn.Module, val_loader, device, scoring):
    model.eval()
    model.to(device)
    val_loss_total = 0.0
    val_n_samples = 0
    preds = []
    trues = []
    with torch.no_grad():
        for val_batch in val_loader:
            batch_x1, batch_x2, batch_y, batch_weight = (t.to(device) for t in val_batch)
            output = model(batch_x1, batch_x2)
            criterion = nn.BCEWithLogitsLoss(batch_weight, reduction='sum')
            val_loss_total += criterion(output['logit'], batch_y.view(-1, 1)).item()
            val_n_samples += len(batch_x1)
            preds.extend(output['sigmoid_output'].view(-1).cpu().numpy())
            trues.extend(batch_y.view(-1).cpu().numpy())
    assert val_n_samples == len(preds)
    val_loss_mean = val_loss_total / val_n_samples
    score = scoring(trues, preds)
    return val_loss_mean, score
