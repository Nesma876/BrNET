"""
BrNet -- Point 4 : détection de duplicats par perceptual hashing
================================================================================
Calcule un hash perceptuel (pHash) pour chaque image du benchmark agrégé
(7023 images, Figshare+SARTAJ+Br35H) et du dataset externe (1137 images,
Mendeley), puis cherche les paires avec une distance de Hamming faible
(quasi-identiques ou identiques), ce qui indiquerait un chevauchement de
source malgré l'absence de documentation officielle.

Pas de GPU nécessaire -- peut tourner sur CPU, rapide (quelques minutes).

Sortie : results_phash_duplicates.csv (paires suspectes, distance <= seuil)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

BENCHMARK_DIR = "/kaggle/input/datasets/rm1000/brain-tumor-mri-scans"
EXTERNAL_DIR = ("/kaggle/input/datasets/souaadrahmoun/brnetval/"
                "Multi-Class Brain Tumor MRI Dataset Glioma, Health/"
                "Brain Cancer MRI Dataset/Brain Cancer MRI Dataset/Brain Cancer MRI Dataset")
HASH_SIZE = 16
DISTANCE_THRESHOLD = 10


def compute_hashes(root_dir, label):
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                img = Image.open(fpath).convert("L")
                h = imagehash.phash(img, hash_size=HASH_SIZE)
                rows.append({"path": fpath, "hash": h, "source": label,
                             "class_folder": os.path.basename(dirpath)})
            except Exception as e:
                print(f"  echec sur {fpath}: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Calcul des pHash -- benchmark agrégé (7023 images) ===")
    df_benchmark = compute_hashes(BENCHMARK_DIR, "benchmark")
    print(f"{len(df_benchmark)} hashes calculés (benchmark)")

    print("\n=== Calcul des pHash -- dataset externe (1137 images) ===")
    df_external = compute_hashes(EXTERNAL_DIR, "external")
    print(f"{len(df_external)} hashes calculés (externe)")

    print(f"\n=== Recherche de paires quasi-identiques (distance <= {DISTANCE_THRESHOLD}) ===")

    suspicious_pairs = []
    external_hashes = df_external["hash"].values
    external_paths = df_external["path"].values
    external_classes = df_external["class_folder"].values

    for i, (bh, bpath) in enumerate(zip(df_benchmark["hash"].values, df_benchmark["path"].values)):
        if i % 1000 == 0:
            print(f"  ... {i}/{len(df_benchmark)} images benchmark comparées")
        distances = np.array([bh - eh for eh in external_hashes])
        close_idx = np.where(distances <= DISTANCE_THRESHOLD)[0]
        for j in close_idx:
            suspicious_pairs.append({
                "benchmark_path": bpath,
                "external_path": external_paths[j],
                "external_class": external_classes[j],
                "hamming_distance": int(distances[j]),
            })

    pairs_df = pd.DataFrame(suspicious_pairs)
    pairs_df.to_csv("results_phash_duplicates.csv", index=False)

    print(f"\n=== RÉSULTAT ===")
    print(f"Paires suspectes trouvées (distance <= {DISTANCE_THRESHOLD}/{HASH_SIZE**2}) : {len(pairs_df)}")
    if len(pairs_df) > 0:
        print(f"Distance minimale trouvée : {pairs_df['hamming_distance'].min()}")
        print(f"Nombre d'images externes distinctes concernées : {pairs_df['external_path'].nunique()}")
        print(f"Nombre d'images benchmark distinctes concernées : {pairs_df['benchmark_path'].nunique()}")
        print("\nDistribution des distances trouvées :")
        print(pairs_df["hamming_distance"].value_counts().sort_index())
    else:
        print("Aucune paire quasi-identique détectée -- pas d'évidence de chevauchement direct.")

    print("\nDétail sauvegardé dans results_phash_duplicates.csv")
