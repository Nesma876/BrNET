"""Reconcile the later audit-v2 tables without model checkpoints."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "audit_v2"


def patient_bootstrap(values, patient_ids, n_boot=2000, seed=123):
    values = np.asarray(values, dtype=float)
    patient_ids = np.asarray(patient_ids)
    patients = np.unique(patient_ids)
    rows = {patient: np.flatnonzero(patient_ids == patient) for patient in patients}
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([rows[patient] for patient in sampled])
        means.append(values[idx].mean())
    return values.mean(), *np.percentile(means, [2.5, 97.5])


def assert_close(actual, expected, atol=5e-5):
    if not np.isclose(actual, expected, atol=atol, rtol=0):
        raise AssertionError(f"{actual} != {expected} (atol={atol})")


def verify_external():
    predictions = pd.read_csv(AUDIT / "results_external_validation_clean.csv")
    manifest = pd.read_csv(AUDIT / "external_all_files_with_exclusion_flag.csv")
    probability_columns = [
        "prob_glioma",
        "prob_healthy",
        "prob_meningioma",
        "prob_pituitary",
    ]
    class_order = np.array(["glioma", "healthy", "meningioma", "pituitary"])
    assert len(predictions) == predictions["path"].nunique() == 452
    assert len(manifest) == manifest["path"].nunique() == 1137
    assert int(manifest["excluded_as_duplicate"].sum()) == 685
    assert not predictions["excluded_as_duplicate"].any()
    assert set(predictions["path"]) == set(
        manifest.loc[~manifest["excluded_as_duplicate"], "path"]
    )
    probabilities = predictions[probability_columns].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=2e-6)
    assert np.array_equal(class_order[probabilities.argmax(axis=1)], predictions["pred_class"])
    correct = int((predictions["true_class"] == predictions["pred_class"]).sum())
    assert correct == 233
    assert_close(correct / len(predictions), 0.5154867257, atol=1e-10)
    assert_close(
        f1_score(predictions["true_class"], predictions["pred_class"], average="macro"),
        0.4235288760,
        atol=1e-10,
    )


def verify_masking():
    masking = pd.read_csv(AUDIT / "results_masking_full_patient_clustered_v2.csv")
    assert len(masking) == masking["image_idx"].nunique() == 2594
    assert masking["patient_id"].nunique() == 220
    assert masking["delta_prob_matched"].isna().sum() == 86
    valid = masking.dropna(subset=["delta_prob_matched"])
    assert len(valid) == 2508
    checks = [
        (masking, masking["changed_tumor"], (0.0625, 0.0438, 0.0841)),
        (masking, masking["delta_prob_tumor"], (0.0380, 0.0261, 0.0524)),
        (valid, valid["changed_matched"], (0.0044, 0.0017, 0.0077)),
        (valid, valid["delta_prob_matched"], (0.0014, 0.0005, 0.0022)),
        (
            valid,
            valid["changed_tumor"] - valid["changed_matched"],
            (0.0578, 0.0398, 0.0790),
        ),
        (
            valid,
            valid["delta_prob_tumor"] - valid["delta_prob_matched"],
            (0.0371, 0.0249, 0.0520),
        ),
    ]
    for frame, values, expected in checks:
        observed = patient_bootstrap(values.to_numpy(), frame["patient_id"].to_numpy())
        for actual, target in zip(observed, expected):
            assert_close(actual, target)


def verify_sanity():
    sample = pd.read_csv(AUDIT / "xai_sanity_check_sample_manifest.csv")
    rows = pd.read_csv(AUDIT / "xai_sanity_check_per_image.csv")
    summary = pd.read_csv(AUDIT / "xai_sanity_check_cascade_summary.csv")
    assert len(sample) == sample["image_idx"].nunique() == 100
    assert len(rows) == 600
    assert set(rows["image_idx"]) == set(sample["image_idx"])
    grouped = rows.groupby("step")["spearman_rho"]
    assert grouped.count().tolist() == [50, 95, 95, 94, 94, 92]
    assert np.allclose(grouped.mean().to_numpy(), summary["mean_spearman_rho"])
    assert np.allclose(grouped.median().to_numpy(), summary["median_spearman_rho"])


def verify_compute():
    summary = pd.read_csv(AUDIT / "compute_benchmark_results.csv").set_index("model")
    for model in ["BrNet", "MobileNetV2", "EfficientNetB7"]:
        latency = pd.read_csv(AUDIT / f"compute_latency_individual_{model}.csv")
        throughput = pd.read_csv(AUDIT / f"compute_throughput_individual_{model}.csv")
        assert len(latency) == 100
        assert len(throughput) == 20
        assert_close(latency["latency_ms"].mean(), summary.loc[model, "latency_mean_ms"], 1e-6)
        assert_close(latency["latency_ms"].median(), summary.loc[model, "latency_median_ms"], 1e-6)
        assert_close(latency["latency_ms"].std(ddof=0), summary.loc[model, "latency_sd_ms"], 1e-6)
        assert_close(
            32 / throughput["batch_time_s"].mean(),
            summary.loc[model, "throughput_images_per_sec"],
            1e-6,
        )


def main():
    verify_external()
    verify_masking()
    verify_sanity()
    verify_compute()
    print("PASS: audit-v2 row-level outputs and raw timings reconciled")


if __name__ == "__main__":
    main()
