"""
BrNet -- McNemar PATIENT-LEVEL (233 patients), réponse point 4 de l'audit Nesma
================================================================================
L'audit signale que le McNemar actuel opère sur 3064 SLICES, pas sur les 233
PATIENTS -- or plusieurs slices viennent du même patient (non independantes),
ce qui peut biaiser le test. Ce script :

  1. Calcule le vote majoritaire par patient (comme deja fait pour BrNet au
     Point 2) pour LES 4 MODELES (BrNet + les 3 baselines patient-disjoint).
  2. Calcule McNemar exact + Holm sur ces 233 PREDICTIONS PAR PATIENT
     (au lieu de 3064 slices), qui devient l'inference principale.

Pas de GPU necessaire -- reutilise les fichiers de predictions deja
calcules.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]

BRNET_FILE = "oof_predictions_full.csv"
MOBILENET_FILE = "oof_predictions_mobilenetv2_v2.csv"
EFFICIENTNET_FILE = "oof_predictions_baselines.csv"
VIT_FILE = "oof_predictions_vit.csv"


def patient_level_vote(df, pred_col, true_col="true_label"):
    rows = []
    for pid, group in df.groupby("patient_id"):
        true_label = group[true_col].iloc[0]
        assert (group[true_col] == true_label).all(), f"Patient {pid}: labels incoherents"

        pred_counts = group[pred_col].value_counts()
        majority_pred = pred_counts.idxmax()
        is_tie = (pred_counts == pred_counts.max()).sum() > 1

        rows.append({
            "patient_id": pid,
            "true_label": true_label,
            "pred_patient_level": majority_pred,
            "n_slices": len(group),
            "was_tie": is_tie,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Chargement des 4 fichiers de predictions slice-level ===")
    df_brnet = pd.read_csv(BRNET_FILE)
    df_mobilenet = pd.read_csv(MOBILENET_FILE)
    df_efficientnet = pd.read_csv(EFFICIENTNET_FILE)
    df_vit = pd.read_csv(VIT_FILE)

    print(f"BrNet: {len(df_brnet)} lignes")
    print(f"MobileNetV2 (corrige): {len(df_mobilenet)} lignes")
    print(f"EfficientNetB7: {len(df_efficientnet)} lignes")
    print(f"ViT-B16: {len(df_vit)} lignes")

    pred_col_brnet = "pred_mean"
    pred_col_mobilenet = "pred_MobileNetV2"
    pred_col_efficientnet = "pred_EfficientNetB7"
    pred_col_vit = "pred_ViT-B16"

    print("\n=== Vote patient-level, les 4 modèles ===")
    pl_brnet = patient_level_vote(df_brnet, pred_col_brnet)
    pl_mobilenet = patient_level_vote(df_mobilenet, pred_col_mobilenet)
    pl_efficientnet = patient_level_vote(df_efficientnet, pred_col_efficientnet)
    pl_vit = patient_level_vote(df_vit, pred_col_vit)

    for name, pl_df in [("BrNet", pl_brnet), ("MobileNetV2", pl_mobilenet),
                          ("EfficientNetB7", pl_efficientnet), ("ViT-B16", pl_vit)]:
        acc = (pl_df["pred_patient_level"] == pl_df["true_label"]).mean()
        print(f"{name}: {len(pl_df)} patients, accuracy patient-level = {acc:.4f}, "
              f"tie-breaks = {pl_df['was_tie'].sum()}")

    print("\n=== Fusion des 4 votes patient-level (INNER JOIN sur patient_id) ===")
    merged = pl_brnet[["patient_id", "true_label", "pred_patient_level"]].rename(
        columns={"pred_patient_level": "pred_BrNet"}
    )
    merged = merged.merge(
        pl_mobilenet[["patient_id", "pred_patient_level"]].rename(
            columns={"pred_patient_level": "pred_MobileNetV2"}), on="patient_id"
    )
    merged = merged.merge(
        pl_efficientnet[["patient_id", "pred_patient_level"]].rename(
            columns={"pred_patient_level": "pred_EfficientNetB7"}), on="patient_id"
    )
    merged = merged.merge(
        pl_vit[["patient_id", "pred_patient_level"]].rename(
            columns={"pred_patient_level": "pred_ViT-B16"}), on="patient_id"
    )

    print(f"Patients apres fusion (doit etre 233) : {len(merged)}")
    assert len(merged) == 233, "ATTENTION: fusion incomplete, verifier les patient_id entre fichiers !"

    merged.to_csv("patient_level_predictions_merged.csv", index=False)

    print("\n=== McNemar exact PATIENT-LEVEL (n=233 patients) ===")
    y_true = merged["true_label"].values
    pred_brnet = merged["pred_BrNet"].values

    results = []
    for baseline_name, baseline_col in [("MobileNetV2", "pred_MobileNetV2"),
                                          ("EfficientNetB7", "pred_EfficientNetB7"),
                                          ("ViT-B16", "pred_ViT-B16")]:
        pred_baseline = merged[baseline_col].values

        brnet_correct = (pred_brnet == y_true)
        baseline_correct = (pred_baseline == y_true)

        n01 = int(((~brnet_correct) & baseline_correct).sum())
        n10 = int((brnet_correct & (~baseline_correct)).sum())

        table = [[0, n01], [n10, 0]]
        result = mcnemar(table, exact=True)

        results.append({
            "comparison": f"BrNet vs {baseline_name}",
            "n01": n01, "n10": n10, "p_exact": result.pvalue,
        })
        print(f"BrNet vs {baseline_name} (patient-level) : n01={n01}, n10={n10}, p_exact={result.pvalue:.4f}")

    pvals = [r["p_exact"] for r in results]
    reject, pvals_holm, _, _ = multipletests(pvals, method="holm")
    for i, r in enumerate(results):
        r["p_holm"] = pvals_holm[i]
        r["significant"] = reject[i]

    print("\n=== Résultats finaux avec correction Holm (PATIENT-LEVEL, n=233) ===")
    for r in results:
        sig = "SIGNIFICATIF" if r["significant"] else "non significatif"
        print(f"{r['comparison']}: n01={r['n01']}, n10={r['n10']}, "
              f"p_exact={r['p_exact']:.4f}, p_holm={r['p_holm']:.4f} [{sig}]")

    print("\nFichier fusionné sauvegardé : patient_level_predictions_merged.csv")
    print("\n>>> Ce résultat (McNemar sur 233 patients) doit maintenant être considéré")
    print(">>> comme l'inférence principale, remplaçant McNemar sur 3064 slices.")
