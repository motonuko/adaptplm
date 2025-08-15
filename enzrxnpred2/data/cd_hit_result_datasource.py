from pathlib import Path

import pandas as pd


# deprecated
def parse_clstr_file(clstr_file: Path):
    clusters = []
    current_cluster = None

    with open(clstr_file, 'r') as file:
        for line in file:
            if line.startswith(">Cluster"):
                if current_cluster is not None:
                    clusters.append(current_cluster)
                cleaned_cluster_id = line.strip()
                if line.startswith('>'):
                    cleaned_cluster_id = cleaned_cluster_id[1:]

                current_cluster = {"cluster_id": cleaned_cluster_id, "sequences": []}
            else:
                parts = line.strip().split()
                # seq_length = parts[1].split('aa,')[0]
                seq_name = parts[2][1:].split("...")[0]
                current_cluster["sequences"].append({"seq_name": seq_name})

        if current_cluster is not None:
            clusters.append(current_cluster)

    return clusters


def _clusters_to_dataframe(clusters):
    data = []
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        for seq in cluster["sequences"]:
            data.append([cluster_id, seq["seq_name"]])

    df = pd.DataFrame(data, columns=["cluster_id", "sequence_name"])
    return df


# deprecated
def load_clstr_file_as_dataframe(ff) -> pd.DataFrame:
    clusters = parse_clstr_file(ff)
    return _clusters_to_dataframe(clusters)


# deprecated: simply use dataframe.
class CdHitResultDatasource:

    def __init__(self, clstr_file: Path, cluster_key: str = 'cluster_id', labels_key: str = 'labels',
                 df_cluster_column_name: str = 'cluster_id', df_label_column_name: str = 'label'):
        self.cluster_key = cluster_key
        self.labels_key = labels_key
        self.cluster_column_name = df_cluster_column_name
        self.label_column_name = df_label_column_name
        self._clusters = self._parse_clstr_file(clstr_file)

    def get_clusters(self):
        return self._clusters

    def get_as_dataframe(self):
        data = []
        for cluster in self._clusters:
            cluster_id = cluster[self.cluster_key]
            for label in cluster[self.labels_key]:
                data.append({self.label_column_name: label, self.cluster_column_name: cluster_id})
        return pd.DataFrame(data)

    def _parse_clstr_file(self, clstr_file: Path):
        clusters = []
        current_cluster = None

        with open(clstr_file, 'r') as file:
            for line in file:
                if line.startswith(">Cluster"):
                    if current_cluster is not None:
                        clusters.append(current_cluster)
                    cleaned_cluster_id = line.strip()
                    if line.startswith('>'):
                        cleaned_cluster_id = cleaned_cluster_id[1:]

                    current_cluster = {self.cluster_key: cleaned_cluster_id, self.labels_key: []}
                else:
                    parts = line.strip().split()
                    # seq_length = parts[1].split('aa,')[0]
                    seq_name = parts[2][1:].split("...")[0]
                    current_cluster[self.labels_key].append(seq_name)

            if current_cluster is not None:
                clusters.append(current_cluster)

        return clusters
