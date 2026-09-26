"""
BrNet -- Point 2 : agrégation patient-level (analyse complémentaire)
================================================================================
Calcule, à partir du fichier out-of-fold déjà produit (oof_predictions_full.csv,
pas de GPU nécessaire ici), une performance au niveau PATIENT en plus du
niveau slice déjà rapporté :
  - Vote majoritaire : la classe prédite la plus fréquente parmi les slices
    d'un patient devient la prédiction "patient-level" pour ce patient.
  - En cas d'égalité, on utilise la moyenne des probabilités softmax
    poolées (déjà disponibles dans prob_mean_c0/c1/c2) comme départage.

Répond explicitement à l'objection du reviewer : "predictions are not
aggregated per patient... unless predictions are aggregated and evaluated
per patient, this should be labelled slice-level performance."

Sortie : results_patient_level_aggregation.csv + résumé imprimé
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

OOF_FILE = "oof_predictions_full.csv"
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]

if __name__ == "__main__":
    df = pd.read_csv(OOF_FILE)
    print(f"=== Chargement : {len(df)} images, {df['patient_id'].nunique()} patients ===")

    prob_cols = ["prob_mean_c0", "prob_mean_c1", "prob_mean_c2"]

    patient_rows = []
    for pid, group in df.groupby("patient_id"):
        true_label = group["true_label"].iloc[0]
        assert (group["true_label"] == true_label).all(), f"Patient {pid} a des labels incohérents !"

        pred_counts = group["pred_mean"].value_counts()
        majority_pred = pred_counts.idxmax()
        is_tie = (pred_counts == pred_counts.max()).sum() > 1

        if is_tie:
            mean_probs = group[prob_cols].mean(axis=0).values
            majority_pred = int(np.argmax(mean_probs))

        patient_rows.append({
            "patient_id": pid,
            "fold": group["fold"].iloc[0],
            "true_label": true_label,
            "true_class": CLASS_NAMES[true_label],
            "n_slices": len(group),
            "pred_patient_level": majority_pred,
            "pred_patient_level_class": CLASS_NAMES[majority_pred],
            "correct_patient_level": majority_pred == true_label,
            "was_tie": is_tie,
        })

    patient_df = pd.DataFrame(patient_rows)
    patient_df.to_csv("results_patient_level_aggregation.csv", index=False)

    acc_patient = patient_df["correct_patient_level"].mean()
    macro_f1_patient = f1_score(patient_df["true_label"], patient_df["pred_patient_level"], average="macro")
    acc_slice = (df["pred_mean"] == df["true_label"]).mean()

    print(f"\n=== RÉSULTATS ===")
    print(f"Nombre de patients : {len(patient_df)}")
    print(f"Nombre de départages par égalité : {patient_df['was_tie'].sum()}")
    print(f"Nombre moyen de slices par patient : {patient_df['n_slices'].mean():.1f} "
          f"(min={patient_df['n_slices'].min()}, max={patient_df['n_slices'].max()})")
    print()
    print(f"Accuracy SLICE-level   (n={len(df)} images)   : {acc_slice:.4f}")
    print(f"Accuracy PATIENT-level (n={len(patient_df)} patients) : {acc_patient:.4f}")
    print(f"Macro-F1 PATIENT-level : {macro_f1_patient:.4f}")

    print("\n=== Accuracy patient-level par fold ===")
    for fold_idx in sorted(patient_df["fold"].unique()):
        sub = patient_df[patient_df["fold"] == fold_idx]
        acc_fold = sub["correct_patient_level"].mean()
        print(f"Fold {int(fold_idx)+1}: n_patients={len(sub)} acc={acc_fold:.4f}")

    print("\n=== Accuracy patient-level par classe ===")
    for c, cname in enumerate(CLASS_NAMES):
        sub = patient_df[patient_df["true_label"] == c]
        acc_c = sub["correct_patient_level"].mean()
        print(f"{cname}: n_patients={len(sub)} acc={acc_c:.4f}")

    print("\nDétail sauvegardé dans results_patient_level_aggregation.csv")
