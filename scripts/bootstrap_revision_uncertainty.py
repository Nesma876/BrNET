"""Recalculate revision-requested uncertainty and pHash sensitivity tables.

The script uses only archived row-level outputs. Patient-clustered intervals
resample patients with replacement and retain every slice belonging to each
sampled patient. Percentile intervals use 10,000 replicates and seed 20260925.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ROW = RESULTS / "row_level"
OUT = RESULTS / "revision_uncertainty"
N_BOOT = 10_000
SEED = 20_260_925


def percentile_ci(values):
    return np.quantile(np.asarray(values, dtype=float), [0.025, 0.975])


def xai_cluster_bootstrap():
    data = pd.read_csv(ROW / "results_xai_full_stratified_v2.csv")
    metrics = ["pointing_hit", "saliency_mass", "iou@0.5"]
    patients = data["patient_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(patients), size=(N_BOOT, len(patients)))
    rows = []

    for correct_label, correct_value in (("Correct", True), ("Incorrect", False)):
        for class_name in ("glioma", "meningioma", "pituitary"):
            subset = data[(data["correct"] == correct_value) & (data["class"] == class_name)]
            patient_group = subset.groupby("patient_id", sort=False)
            for metric in metrics:
                sums = patient_group[metric].sum().reindex(patients, fill_value=0).to_numpy(float)
                counts = patient_group[metric].count().reindex(patients, fill_value=0).to_numpy(float)
                boot = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
                low, high = percentile_ci(boot)
                rows.append({
                    "stratum": correct_label,
                    "class": class_name.capitalize(),
                    "n_slices": len(subset),
                    "n_patients": subset["patient_id"].nunique(),
                    "metric": metric,
                    "estimate": subset[metric].mean(),
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_unit": "patient",
                    "n_bootstrap": N_BOOT,
                    "seed": SEED,
                })
    return pd.DataFrame(rows)


def patient_accuracy_difference_bootstrap():
    brnet = pd.read_csv(ROW / "oof_predictions_full.csv")
    baselines = pd.read_csv(ROW / "oof_predictions_baselines.csv")
    mobile = pd.read_csv(ROW / "oof_predictions_mobilenetv2_v2.csv")
    vit = pd.read_csv(ROW / "oof_predictions_vit.csv")
    data = brnet[["image_idx", "patient_id", "true_label", "pred_mean",
                  "prob_mean_c0", "prob_mean_c1", "prob_mean_c2"]].merge(
        mobile[["image_idx", "pred_MobileNetV2", "prob_MobileNetV2_c0",
                "prob_MobileNetV2_c1", "prob_MobileNetV2_c2"]],
        on="image_idx", validate="one_to_one",
    ).merge(
        baselines[["image_idx", "pred_EfficientNetB7", "prob_EfficientNetB7_c0",
                   "prob_EfficientNetB7_c1", "prob_EfficientNetB7_c2"]],
        on="image_idx", validate="one_to_one",
    ).merge(
        vit[["image_idx", "pred_ViT-B16", "prob_ViT-B16_c0", "prob_ViT-B16_c1",
             "prob_ViT-B16_c2"]], on="image_idx", validate="one_to_one",
    )

    def aggregate_patient(group, pred_col, prob_prefix):
        counts = group[pred_col].value_counts()
        tied = counts[counts == counts.max()].index.to_numpy(dtype=int)
        if len(tied) == 1:
            return int(tied[0])
        mean_prob = group[[f"{prob_prefix}_c0", f"{prob_prefix}_c1", f"{prob_prefix}_c2"]].mean().to_numpy()
        return int(tied[np.argmax(mean_prob[tied])])

    specs = {
        "BrNet": ("pred_mean", "prob_mean"),
        "MobileNetV2": ("pred_MobileNetV2", "prob_MobileNetV2"),
        "EfficientNetB7": ("pred_EfficientNetB7", "prob_EfficientNetB7"),
        "ViT-B/16": ("pred_ViT-B16", "prob_ViT-B16"),
    }
    patient_rows = []
    for patient_id, group in data.groupby("patient_id", sort=False):
        row = {"patient_id": patient_id, "true_label": int(group["true_label"].mode().iat[0])}
        for model, (pred_col, prob_prefix) in specs.items():
            row[model] = aggregate_patient(group, pred_col, prob_prefix)
        patient_rows.append(row)
    data = pd.DataFrame(patient_rows)
    true = data["true_label"].to_numpy()
    brnet_ok = data["BrNet"].to_numpy() == true
    comparisons = {
        "MobileNetV2": "MobileNetV2",
        "EfficientNetB7": "EfficientNetB7",
        "ViT-B/16": "ViT-B/16",
    }
    rng = np.random.default_rng(SEED)
    sampled = rng.integers(0, len(data), size=(N_BOOT, len(data)))
    rows = []
    for model, column in comparisons.items():
        baseline_ok = data[column].to_numpy() == true
        per_patient_difference = brnet_ok.astype(float) - baseline_ok.astype(float)
        boot = per_patient_difference[sampled].mean(axis=1)
        low, high = percentile_ci(boot)
        rows.append({
            "comparison": f"BrNet - {model}",
            "n_patients": len(data),
            "brnet_accuracy": brnet_ok.mean(),
            "baseline_accuracy": baseline_ok.mean(),
            "accuracy_difference": per_patient_difference.mean(),
            "ci95_low": low,
            "ci95_high": high,
            "bootstrap_unit": "patient",
            "n_bootstrap": N_BOOT,
            "seed": SEED,
        })
    return pd.DataFrame(rows)


def phash_threshold_sensitivity():
    data = pd.read_csv(ROW / "results_phash_duplicates.csv")
    rows = []
    for threshold in (0, 5, 10):
        subset = data[data["hamming_distance"] <= threshold]
        rows.append({
            "threshold": threshold,
            "hash_bits": 256,
            "match_pairs": len(subset),
            "unique_external_images": subset["external_path"].nunique(),
            "total_external_images": 1137,
            "external_images_percent": 100 * subset["external_path"].nunique() / 1137,
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "xai_patient_clustered_bootstrap.csv": xai_cluster_bootstrap(),
        "patient_accuracy_difference_bootstrap.csv": patient_accuracy_difference_bootstrap(),
        "phash_threshold_sensitivity.csv": phash_threshold_sensitivity(),
    }
    patient = outputs["patient_accuracy_difference_bootstrap.csv"]
    expected = np.array([0.8755364806866953, 0.8969957081545065, 0.9012875536480687])
    assert np.allclose(patient["brnet_accuracy"], 0.8626609442060086)
    assert np.allclose(patient["baseline_accuracy"], expected)
    phash = outputs["phash_threshold_sensitivity.csv"]
    assert phash["match_pairs"].tolist() == [466, 813, 880]
    assert phash["unique_external_images"].tolist() == [433, 681, 685]
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)
        print(f"wrote {OUT / name} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
