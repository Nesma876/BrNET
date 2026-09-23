"""Recompute the frozen OOF comparison directly from published row-level CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "row_level"
FROZEN = ROOT / "results" / "FROZEN_MODEL_COMPARISON.csv"


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate(pvalues[order] * np.arange(len(pvalues), 0, -1)),
    )
    return adjusted


def mcnemar_counts(y_true, brnet_pred, baseline_pred):
    a = np.asarray(brnet_pred) == np.asarray(y_true)
    b = np.asarray(baseline_pred) == np.asarray(y_true)
    n00 = int((~a & ~b).sum())
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n11 = int((a & b).sum())
    pvalue = float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return n00, n01, n10, n11, pvalue


def patient_predictions(base, prediction, probabilities):
    work = base[["patient_id", "true_label"]].copy()
    work["prediction"] = np.asarray(prediction)
    rows = []
    for patient_id, group in work.groupby("patient_id", sort=True):
        indices = group.index.to_numpy()
        counts = group.prediction.value_counts()
        tied = counts[counts == counts.max()].index.to_numpy()
        if len(tied) == 1:
            pred = int(tied[0])
        else:
            pred = int(np.asarray(probabilities)[indices].mean(axis=0).argmax())
        rows.append((patient_id, int(group.true_label.iloc[0]), pred))
    return pd.DataFrame(rows, columns=["patient_id", "true_label", "prediction"])


def main():
    brnet = pd.read_csv(DATA / "oof_predictions_full.csv").sort_values("image_idx").reset_index(drop=True)
    mobile = pd.read_csv(DATA / "oof_predictions_mobilenetv2_v2.csv").sort_values("image_idx").reset_index(drop=True)
    baselines = pd.read_csv(DATA / "oof_predictions_baselines.csv").sort_values("image_idx").reset_index(drop=True)
    vit = pd.read_csv(DATA / "oof_predictions_vit.csv").sort_values("image_idx").reset_index(drop=True)
    keys = ["image_idx", "patient_id", "fold", "true_label", "true_class"]
    assert brnet[keys].equals(mobile[keys])
    assert brnet[keys].equals(baselines[keys])
    assert brnet[keys].equals(vit[keys])
    assert len(brnet) == 3064 and brnet.patient_id.nunique() == 233
    assert brnet.groupby("patient_id").fold.nunique().max() == 1
    assert brnet.groupby("patient_id").true_label.nunique().max() == 1

    models = {
        "BrNet": (
            brnet.pred_mean.to_numpy(),
            brnet[[f"prob_mean_c{i}" for i in range(3)]].to_numpy(),
        ),
        "MobileNetV2 corrected": (
            mobile.pred_MobileNetV2.to_numpy(),
            mobile[[f"prob_MobileNetV2_c{i}" for i in range(3)]].to_numpy(),
        ),
        "EfficientNetB7": (
            baselines.pred_EfficientNetB7.to_numpy(),
            baselines[[f"prob_EfficientNetB7_c{i}" for i in range(3)]].to_numpy(),
        ),
        "ViT-B/16": (
            vit["pred_ViT-B16"].to_numpy(),
            vit[[f"prob_ViT-B16_c{i}" for i in range(3)]].to_numpy(),
        ),
    }
    y_true = brnet.true_label.to_numpy()
    patients = {name: patient_predictions(brnet, pred, prob) for name, (pred, prob) in models.items()}
    frozen = pd.read_csv(FROZEN)

    for level in ("slice", "patient"):
        for name, (pred, _) in models.items():
            observed = float((pred == y_true).mean()) if level == "slice" else float((patients[name].prediction == patients[name].true_label).mean())
            expected = float(frozen.loc[(frozen.analysis_level == level) & (frozen.model_or_comparison == name), "accuracy"].iloc[0])
            assert np.isclose(observed, expected, atol=1e-15), (level, name, observed, expected)

        raw = []
        calculated = []
        for name in list(models)[1:]:
            if level == "slice":
                values = mcnemar_counts(y_true, models["BrNet"][0], models[name][0])
            else:
                values = mcnemar_counts(patients["BrNet"].true_label, patients["BrNet"].prediction, patients[name].prediction)
            raw.append(values[4])
            calculated.append((name, values))
        adjusted = holm(raw)
        for (name, values), p_holm in zip(calculated, adjusted):
            expected = frozen.loc[(frozen.analysis_level == level) & (frozen.model_or_comparison == f"BrNet vs {name}")].iloc[0]
            assert tuple(values[:4]) == tuple(int(expected[c]) for c in ("n00", "n01", "n10", "n11"))
            assert np.isclose(values[4], float(expected.exact_p), atol=1e-15)
            assert np.isclose(p_holm, float(expected.holm_adjusted_p), atol=1e-15)

    print("PASS: frozen model comparison reproduced from published row-level OOF predictions")


if __name__ == "__main__":
    main()
