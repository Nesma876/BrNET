"""
BrNet -- pHash par partition train/val/test (Point 12 de l'audit Nesma)
================================================================================
Le résultat déjà publié (60.2% du dataset externe contaminé) compare contre
le benchmark agrégé COMPLET (7023 images), sans préciser si les doublons
détectés tombent dans le TRAIN (le modèle les a vus à l'entraînement, donc
mémorisation possible), le VAL, ou le TEST (aucun impact sur l'entraînement).
Ce script résout cette ambiguïté en croisant chaque paire détectée avec le
split 80/10/10 déterministe (seed=12) déjà utilisé partout ailleurs.

Nécessite : results_phash_duplicates.csv (déjà généré, Point 4).
Pas de GPU nécessaire.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/datasets/rm1000/brain-tumor-mri-scans"
DUPLICATES_FILE = "/kaggle/input/datasets/souaadrahmoun/lastupload/results_phash_duplicates.csv"
SPLIT_SEED = 12


def build_file_list():
    class_names = sorted(os.listdir(DATA_DIR))
    filepaths, labels = [], []
    for label_idx, cls in enumerate(class_names):
        cls_dir = os.path.join(DATA_DIR, cls)
        files = sorted(os.listdir(cls_dir))
        for fname in files:
            filepaths.append(os.path.join(cls_dir, fname))
            labels.append(label_idx)
    return np.array(filepaths), np.array(labels), class_names


def split_indices(n_total, seed=SPLIT_SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    train_size = int(0.8 * n_total)
    val_size = int(0.1 * n_total)
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]
    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    print("=== Étape 1 : reconstruire le split 80/10/10 déterministe ===")
    filepaths, labels, class_names = build_file_list()
    n_total = len(filepaths)
    print(f"Total images benchmark agrégé : {n_total}")

    train_idx, val_idx, test_idx = split_indices(n_total)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    partition_map = {}
    for i in train_idx:
        partition_map[filepaths[i]] = "train"
    for i in val_idx:
        partition_map[filepaths[i]] = "val"
    for i in test_idx:
        partition_map[filepaths[i]] = "test"

    print("\n=== Étape 2 : croiser avec les paires détectées par pHash ===")
    dup_df = pd.read_csv(DUPLICATES_FILE)
    print(f"Paires de quasi-duplicats chargées : {len(dup_df)}")

    dup_df["partition"] = dup_df["benchmark_path"].map(partition_map)

    n_unmatched = dup_df["partition"].isna().sum()
    if n_unmatched > 0:
        print(f"ATTENTION : {n_unmatched} paires n'ont pas pu être associées à une partition "
              f"(chemin non trouvé -- vérifier que DATA_DIR correspond exactement au "
              f"dataset utilisé pour le benchmark agrégé).")

    print("\n=== Répartition des quasi-duplicats par partition (paires) ===")
    print(dup_df["partition"].value_counts())

    print("\n=== Répartition par partition, EN IMAGES EXTERNES DISTINCTES ===")
    ext_partition = dup_df.groupby("external_path")["partition"].apply(
        lambda s: sorted(set(s.dropna()))
    ).reset_index()
    ext_partition["partition_str"] = ext_partition["partition"].apply(lambda x: "+".join(x) if x else "unmatched")

    print(ext_partition["partition_str"].value_counts())

    n_in_train = ext_partition["partition_str"].str.contains("train").sum()
    n_in_val = ext_partition["partition_str"].str.contains("val").sum()
    n_in_test = ext_partition["partition_str"].str.contains("test").sum()
    n_total_contaminated = len(ext_partition)

    print(f"\n=== Résumé (sur {n_total_contaminated} images externes contaminées au total) ===")
    print(f"Avec un doublon en TRAIN : {n_in_train} ({n_in_train/n_total_contaminated*100:.1f}%)")
    print(f"Avec un doublon en VAL   : {n_in_val} ({n_in_val/n_total_contaminated*100:.1f}%)")
    print(f"Avec un doublon en TEST  : {n_in_test} ({n_in_test/n_total_contaminated*100:.1f}%)")

    ext_partition.to_csv("phash_partition_breakdown.csv", index=False)
    dup_df.to_csv("results_phash_duplicates_with_partition.csv", index=False)
    print("\nFichiers sauvegardés : phash_partition_breakdown.csv, "
          "results_phash_duplicates_with_partition.csv")
