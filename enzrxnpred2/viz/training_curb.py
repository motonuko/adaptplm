from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from adaptplm.core.default_path import DefaultPath
from adaptplm.extension.matplotlib_ext import load_font_ipaexg


def export_multiple_training_curves_to_pdf(
        log_dirs: List[Path],
        metric_name,
        ax,
        labels=None,
        smoothing_window=1,
        y_label: str = ''
):
    load_font_ipaexg()
    fofofo = 'IPAexGothic'

    if labels and len(labels) != len(log_dirs):
        raise ValueError("The length of `labels` must match the length of `log_dirs`.")

    # Default labels
    # if labels is None:
    #     labels = [f"Run {i + 1}" for i in range(len(log_dirs))]

    # plt.figure(figsize=(10, 6))

    for log_dir, label in zip(log_dirs, labels):
        # Load TensorBoard data
        event_acc = EventAccumulator(log_dir.as_posix())
        event_acc.Reload()

        # Retrieve data for the specified metric name
        if metric_name not in event_acc.Tags()['scalars']:
            print(f"Warning: Metric '{metric_name}' not found in directory '{log_dir.as_posix()}'. Skipping.")
            continue

        events = event_acc.Scalars(metric_name)
        data = [(e.step, e.value) for e in events]

        df = pd.DataFrame(data, columns=["Step", "Value"])

        # Smoothing
        if smoothing_window > 1:
            df["Smoothed"] = df["Value"].rolling(window=smoothing_window, center=True).mean()
        else:
            df["Smoothed"] = df["Value"]

        # Plotting
        ax.plot(df["Step"], df["Smoothed"], label=label, linewidth=2)

    # Set format for the figures
    #
    # plt.title(f"Training Curves: {metric_name}", fontsize=16)
    # plt.figure(figsize=(8, 6))
    ax.set_xlabel("ステップ", fontdict={'fontname': fofofo, 'fontsize': 10})
    ax.set_ylabel(y_label, fontdict={'fontname': fofofo, 'fontsize': 10})
    ax.set_ylim(0, 0.15)
    if not all([l is None for l in labels]):
        ax.legend(fontsize=10)
    ax.grid(True)


def draw_two_in_one(
        log_dirs: List[Path],
        metric_name1,
        metric_name2,
        pdf_output_path,
        y_label1: str = '',
        y_label2: str = '',
        labels=None,
        smoothing_window=1,

):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))

    export_multiple_training_curves_to_pdf(
        log_dirs,
        metric_name1,
        axs[0],
        labels=labels,
        y_label=y_label1
    )

    export_multiple_training_curves_to_pdf(
        log_dirs,
        metric_name2,
        axs[1],
        labels=labels,
        y_label=y_label2
    )

    # レイアウトを調整
    plt.tight_layout()
    # plt.show()
    plt.savefig(pdf_output_path, format="pdf", dpi=300)
    plt.close()


if __name__ == '__main__':
    # RXN
    log_dirs = [
        DefaultPath().data_dir / 'server_events' / 'exp' / 'rxn_encoder_train' / '241218_012453',
        # DefaultPath().data_dir / 'server_events' / 'exp' / 'rxn_encoder_train' / '250117_103744',
    ]
    output_dir = DefaultPath().build / 'fig' / 'train_curb'
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_two_in_one(
        log_dirs,
        metric_name1="PerEvalStep/TrainingMeanCumulativeLoss",
        metric_name2="PerEvalStep/ValMeanLoss",
        pdf_output_path=output_dir / f"rxn_loss_curves.pdf",
        # labels=['trial 1'],
        labels=[None],
        # labels=['trial 1', 'trial 2'],
        # y_label1='Training Loss',
        # y_label2='Val. Loss',
        y_label1='訓練損失',
        y_label2='検証損失'
    )

    # SEQ_RXN
    log_base = DefaultPath().data_dir / 'server_events' / 'exp' / 'seqrxn_encoder_train'

    log_dirs = [
        log_base / '241229_023204',  # mlm 0.3
        log_base / '241226_164549',  # No pretraining
        # log_base / '250101_123329',  # 2 additional layer
    ]
    output_dir = DefaultPath().build / 'fig' / 'train_curb'
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_two_in_one(
        log_dirs,
        metric_name1="PerEvalStep/TrainingMeanCumulativeLoss",
        metric_name2="PerEvalStep/ValMeanLoss",
        pdf_output_path=output_dir / f"seqrxn_loss_curves.pdf",
        # labels=[
        #     'pretrained',
        #     "w/o pretraining",
        #     # '2 additional layers'
        # ],
        labels=[
            '事前学習済み',
            "事前学習なし",
        ],
        # y_label1='Training Loss',
        # y_label2='Val. Loss'
        y_label1='訓練損失',
        y_label2='検証損失'
    )
    # export_multiple_training_curves_to_pdf(
    #     log_dirs,
    # )

    # RXN
    # log_base = DefaultPath().data_dir / 'server_events' / 'exp' / 'seqrxn_encoder_train'
    # log_dirs = [
    #     log_base / '241229_023204',  # mlm 0.3
    #     log_base / '241226_164549',  # No pretraining
    #     log_base / '250101_123329',  # 2 additional layer
    # ]
    # output_dir = DefaultPath().build / 'fig' / 'train_curb'
    # output_dir.mkdir(parents=True, exist_ok=True)
    # export_multiple_training_curves_to_pdf(
    #     log_dirs,
    #     pdf_output_path=output_dir / f"seqrxn_val_loss_curves.pdf",
    #     labels=['base', "No pretraining", '2 additional layers'],
    # )
