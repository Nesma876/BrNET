"""
BrNet -- Calibration (Point 13 de l'audit Nesma)
================================================================================
Calcule ECE, Brier score, reliability curve, et sensibilité/spécificité/
PPV/NPV par classe. Pas de GPU nécessaire -- réutilise les probabilités
déjà sauvegardées dans oof_predictions_full.csv (patient-disjoint).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OOF_FILE = "oof_predictions_full.csv"
N_CLASSES = 3
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
N_BINS = 10


def expected_calibration_error(y_true, y_prob, n_bins=N_BINS):
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_data.append({
                "bin_lo": lo, "bin_hi": hi, "n": in_bin.sum(),
                "accuracy": acc_in_bin, "confidence": conf_in_bin,
            })
        else:
            bin_data.append({"bin_lo": lo, "bin_hi": hi, "n": 0, "accuracy": np.nan, "confidence": np.nan})

    return ece, pd.DataFrame(bin_data)


def multiclass_brier_score(y_true, y_prob, n_classes):
    y_onehot = np.eye(n_classes)[y_true]
    return np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1))


def per_class_sens_spec_ppv_npv(y_true, y_pred, n_classes, class_names):
    rows = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        tn = np.sum((y_true != c) & (y_pred != c))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan

        rows.append({
            "class": class_names[c], "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "sensitivity": sensitivity, "specificity": specificity,
            "PPV": ppv, "NPV": npv,
        })
    return pd.DataFrame(rows)


def plot_reliability_diagram(bin_df, title, output_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    valid = bin_df.dropna(subset=["accuracy"])
    bin_centers = (valid["bin_lo"] + valid["bin_hi"]) / 2

    ax.bar(bin_centers, valid["accuracy"], width=1.0 / N_BINS * 0.9,
           edgecolor="black", alpha=0.7, label="Accuracy observée")
    ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")
    ax.set_xlabel("Confiance (probabilité prédite)")
    ax.set_ylabel("Accuracy observée")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Diagramme de fiabilité sauvegardé : {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=== Chargement des prédictions patient-disjoint (BrNet) ===")
    df = pd.read_csv(OOF_FILE)
    y_true = df["true_label"].values
    y_prob = df[["prob_mean_c0", "prob_mean_c1", "prob_mean_c2"]].values
    y_pred = df["pred_mean"].values

    print(f"{len(df)} images")

    print("\n=== ECE (Expected Calibration Error) ===")
    ece, bin_df = expected_calibration_error(y_true, y_prob, N_BINS)
    print(f"ECE = {ece:.4f}")
    print("\nDétail par bin de confiance :")
    print(bin_df.to_string(index=False))

    print("\n=== Brier Score (multiclasse) ===")
    brier = multiclass_brier_score(y_true, y_prob, N_CLASSES)
    print(f"Brier score = {brier:.4f} (0 = parfait; max théorique = 2)")

    print("\n=== Sensibilité / Spécificité / PPV / NPV par classe ===")
    metrics_df = per_class_sens_spec_ppv_npv(y_true, y_pred, N_CLASSES, CLASS_NAMES)
    print(metrics_df.to_string(index=False))

    print("\n=== Génération du diagramme de fiabilité ===")
    plot_reliability_diagram(
        bin_df, "BrNet -- Diagramme de fiabilité (patient-disjoint)",
        "reliability_diagram_brnet_patient_disjoint.png"
    )

    bin_df.to_csv("calibration_bins_brnet.csv", index=False)
    metrics_df.to_csv("clinical_metrics_brnet.csv", index=False)

    print("\n=== RÉSUMÉ ===")
    print(f"ECE : {ece:.4f}")
    print(f"Brier score : {brier:.4f}")
    print("Fichiers sauvegardés : calibration_bins_brnet.csv, "
          "clinical_metrics_brnet.csv, reliability_diagram_brnet_patient_disjoint.png")
