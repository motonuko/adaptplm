import re
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.data.seq_embedding_data import SequenceEmbeddingData


def compute_cluster_silhouette_scores(X, labels):
    sample_silhouette_values = silhouette_samples(X, labels)
    unique_labels = np.unique(labels)
    cluster_scores = {}

    for label in unique_labels:
        cluster_values = sample_silhouette_values[labels == label]
        cluster_scores[label] = cluster_values.mean()

    return cluster_scores


def compute_overall_mean_silhouette_score(X, labels):
    cluster_scores = compute_cluster_silhouette_scores(X, labels)
    mean_score = np.mean(list(cluster_scores.values()))
    return mean_score


def plot_cluster_silhouette_distributions(X, labels):
    sample_silhouette_values = silhouette_samples(X, labels)
    unique_labels = np.unique(labels)

    # create list for each cluster
    data = [sample_silhouette_values[labels == label] for label in unique_labels]

    # create position index
    positions = np.arange(len(unique_labels)) + 1

    plt.figure(figsize=(20, 6))
    plt.boxplot(data, positions=positions, widths=0.6)

    plt.xlabel("Cluster Label")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Distribution per Cluster")
    plt.xticks(positions, unique_labels, rotation=45)
    plt.ylim(-1, 1)  # the range of silhouette
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.show()


def c_index(X, labels):
    intra_distances = []
    inter_distances = []

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = np.linalg.norm(X[i] - X[j])
            if labels[i] == labels[j]:
                intra_distances.append(d)
            else:
                inter_distances.append(d)

    min_inter = min(inter_distances)
    max_intra = max(intra_distances)

    return (sum(intra_distances) - min_inter) / (max_intra - min_inter) if max_intra > min_inter else np.inf


#
# def vrc(X, labels):
#     unique_labels = np.unique(labels)
#     n_clusters = len(unique_labels)
#     overall_mean = np.mean(X, axis=0)
#
#     between_variance = sum(len(X[labels == label]) * np.linalg.norm(X[labels == label].mean(axis=0) - overall_mean) ** 2
#                            for label in unique_labels)
#
#     within_variance = sum(np.sum((X[labels == label] - X[labels == label].mean(axis=0)) ** 2)
#                           for label in unique_labels)
#
#     return (between_variance / (n_clusters - 1)) / (within_variance / (len(X) - n_clusters)) if len(
#         X) > n_clusters else np.inf

# vrc_score = vrc(np.vstack(df["embedding"].values), df[key].values)
# print("Variance Ratio Criterion:", vrc_score)


def mean_cluster_cosine_similarity(X, labels):
    unique_labels = np.unique(labels)
    similarities = []

    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            sim_matrix = cosine_similarity(cluster_points)
            upper_triangular = sim_matrix[np.triu_indices(len(cluster_points), k=1)]
            similarities.append(np.mean(upper_triangular))

    return np.mean(similarities) if similarities else 0


def mean_inter_cluster_distance(X, labels):
    unique_labels = np.unique(labels)
    cluster_centers = np.array([X[labels == label].mean(axis=0) for label in unique_labels])

    # Compute distance between clusters
    inter_cluster_distances = cdist(cluster_centers, cluster_centers)
    np.fill_diagonal(inter_cluster_distances, np.nan)  # Remove distances between same cluster.
    return np.nanmean(inter_cluster_distances)


class ECLevel(Enum):
    EC_CLASS = "ec_class"
    EC_SUBCLASS = "ec_subclass"
    EC_SUB_SUBCLASS = "ec_sub_subclass"


def draw(seq_ec_file: Path,
         embedding_npy_file: Path,
         key=ECLevel.EC_SUB_SUBCLASS.value):
    df_seq_ec = pd.read_csv(seq_ec_file, dtype={'ec_number': str, 'ec_class': str})
    df_seq_ec = df_seq_ec.dropna()
    df_embeddings = SequenceEmbeddingData.from_npy(embedding_npy_file).df
    df = df_seq_ec.merge(df_embeddings, on='sequence')
    df = df.dropna()

    df[ECLevel.EC_SUBCLASS.value] = df['ec_number'].apply(lambda x: '.'.join(x.split('.')[:2]))
    df[ECLevel.EC_SUB_SUBCLASS.value] = df['ec_number'].apply(lambda x: '.'.join(x.split('.')[:3]))

    min_class_size = 2
    class_counts = df[key].value_counts()
    print(class_counts)
    small_classes = class_counts[class_counts < min_class_size].index
    print(len(small_classes))
    df = df[~df[key].isin(small_classes)]

    class_counts = df['ec_class'].value_counts()
    print(class_counts)
    print(f"total pairs: {len(df)}")
    print(f"total clases: {len(df[key].unique())}")

    # df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(str(x)))
    X = np.vstack(df["embedding"].values)

    labels = df[key].values

    # plot_cluster_silhouette_distributions(X, labels)
    results = {
        # "Silhouette Score": silhouette_score(X, labels),
        "DB Index": davies_bouldin_score(X, labels),
        "CH Index": calinski_harabasz_score(X, labels),
        "Mean Silhouette (Cluster)": compute_overall_mean_silhouette_score(X, labels),
        # "C-Index": c_index(np.vstack(df["embedding"].values), df[key].values),
        # "Mean Cluster Cosine Similarity": mean_cluster_cosine_similarity(np.vstack(df["embedding"].values),
        #                                                                  df[key].values),
        # "Mean Inter-Cluster Distance": mean_inter_cluster_distance(X, labels)
    }
    return results


def get_label(file_stem: str):
    if re.match(r'^embeddings_for_embeddings_evaluation_\d{6}_\d{6}$', file_stem):
        return "ESM-1b$_\\mathrm{{DA}}$"
    elif re.match(r'^embeddings_for_embeddings_evaluation_esm1b_t33_650M_UR50S$', file_stem):
        return "ESM-1b$_\\mathrm{{MEAN}}$"
    else:
        return file_stem.replace('embeddings_for_embeddings_evaluation_', '').replace('_', '\_')


def compute_clustering_scores(seq_ec_file_path: str, embedding_files: List[str]):
    results = []
    names = []
    embed_files = [Path(embed) for embed in embedding_files]
    for embed_file in embed_files:
        results.append(draw(seq_ec_file=Path(seq_ec_file_path), embedding_npy_file=embed_file))
        # names.append(embed_file.stem.replace('embeddings_for_embeddings_evaluation_', '').replace('_', '\_'))
        names.append(get_label(embed_file.stem))
    df = pd.DataFrame(results, index=names)

    def format_sigfig(x, sig=3):
        """Round to 'sig' significant figures"""
        if pd.isna(x):  # support NaN
            return ""
        return f"{x:.{sig}g}"

    df = df.apply(lambda col: col.map(format_sigfig))
    df.to_csv(embed_files[0].parent / 'summary.csv')
    df.to_latex(embed_files[0].parent / 'summary.tex')


#
if __name__ == '__main__':
    target_dir = DefaultPath().build / 'embed'
    embed_files = [
        target_dir / 'embeddings_for_embeddings_evaluation_esm1b_t33_650M_UR50S.npz',
        target_dir / 'embeddings_for_embeddings_evaluation_250420_121652.npz',
    ]
    compute_clustering_scores(target_dir / 'embedding_evaluation_seq_ec.csv',
                              embed_files)
