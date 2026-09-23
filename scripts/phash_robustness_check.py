"""
BrNet -- Tests de robustesse complémentaires sur la détection pHash
================================================================================
1. Sensibilité au seuil : combien d'images sont marquées "contaminées" à
   différents seuils de distance de Hamming (0 à 10) ? Le résultat principal
   (60.2% contaminé à seuil=10) est-il stable autour de ce choix ?
2. Répartition des contaminations par classe externe : est-ce que certaines
   classes (ex. pituitary) sont bien plus contaminées que d'autres, ce qui
   expliquerait le déséquilibre observé dans le sous-ensemble propre ?

Nécessite : results_phash_duplicates.csv (Point 4, déjà généré).
"""

import os
import pandas as pd

DUPLICATES_FILE = "results_phash_duplicates.csv"
EXTERNAL_DIR = ("/kaggle/input/datasets/souaadrahmoun/brnetval/"
                "Multi-Class Brain Tumor MRI Dataset Glioma, Health/"
                "Brain Cancer MRI Dataset/Brain Cancer MRI Dataset/Brain Cancer MRI Dataset")

CLASS_MAPPING = {
    "Glioma": "glioma",
    "Healthy": "healthy",
    "Meningioma": "meningioma",
    "Pituitary Macroadenoma": "pituitary",
}

if __name__ == "__main__":
    dup_df = pd.read_csv(DUPLICATES_FILE)
    print(f"Fichier chargé : {len(dup_df)} paires (toutes à distance <= 10, seuil déjà appliqué)")

    print("\n=== TEST 1 : Nombre d'images externes contaminées selon le seuil ===")
    print("(Seuils > 10 nécessiteraient un nouveau calcul complet des distances -- non fait ici)")
    for threshold in [0, 2, 4, 6, 8, 10]:
        contaminated = dup_df[dup_df["hamming_distance"] <= threshold]["external_path"].nunique()
        print(f"  Seuil <= {threshold:>2} : {contaminated} images externes contaminées "
              f"({contaminated/1137*100:.1f}% du dataset externe)")

    print("\n=== TEST 2 : Répartition des images contaminées par classe externe ===")

    total_by_class = {}
    for cls_folder, cls_name in CLASS_MAPPING.items():
        folder = os.path.join(EXTERNAL_DIR, cls_folder)
        if os.path.isdir(folder):
            total_by_class[cls_name] = len(os.listdir(folder))

    contaminated_by_class = dup_df.drop_duplicates("external_path")["external_class"].value_counts()
    contaminated_by_class_named = {}
    for folder_name, count in contaminated_by_class.items():
        clean_name = CLASS_MAPPING.get(folder_name, folder_name)
        contaminated_by_class_named[clean_name] = count

    print(f"\n{'Classe':<12}{'Total':<10}{'Contaminées':<15}{'% contaminé':<12}{'Propres':<10}")
    for cls_name, total in total_by_class.items():
        contam = contaminated_by_class_named.get(cls_name, 0)
        pct = contam / total * 100 if total > 0 else 0
        clean = total - contam
        print(f"{cls_name:<12}{total:<10}{contam:<15}{pct:<12.1f}{clean:<10}")

    print("\n=== Interprétation ===")
    max_class = max(contaminated_by_class_named, key=lambda c: contaminated_by_class_named[c] / total_by_class[c])
    print(f"Classe la plus contaminée (en proportion) : {max_class}")
    print("Ceci explique directement pourquoi cette classe est sous-représentée")
    print("dans le sous-ensemble 'propre' utilisé pour la validation externe corrigée.")
