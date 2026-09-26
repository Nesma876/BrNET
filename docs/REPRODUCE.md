# Reproducing the current numerical results

## One-command checks

```bash
python -m pytest -q
python scripts/verify_frozen_results.py
python scripts/bootstrap_revision_uncertainty.py
```

`verify_frozen_results.py` reads only the four canonical OOF prediction tables and `results/FROZEN_MODEL_COMPARISON.csv`. It verifies row alignment, patient-disjoint folds, slice and patient accuracies, discordant-pair cells, exact McNemar p-values, and Holm correction.

`bootstrap_revision_uncertainty.py` reads the canonical OOF, pHash, and Grad-CAM row-level tables. With 10,000 resamples and seed 20260925, it regenerates the three tables in `results/revision_uncertainty/`.

## Source-to-result map

| Reported result | Primary source |
|---|---|
| Four-model OOF performance | `results/row_level/oof_predictions_*.csv` |
| Patient-level McNemar and Holm tests | OOF tables plus `scripts/verify_frozen_results.py` |
| Accuracy-difference confidence intervals | OOF tables plus `scripts/bootstrap_revision_uncertainty.py` |
| pHash overlap and sensitivity | `results/row_level/results_phash_duplicates.csv` |
| Partition overlap counts | `results/row_level/results_phash_partition_summary.csv` |
| Grad-CAM localization and clustered intervals | `results/row_level/results_xai_full_stratified_v2.csv` |
| Fixed patient folds | `data/manifests/figshare_patient_fold_manifest.csv` |

The experiment scripts require the public datasets and, where applicable, GPU resources. They document the implemented preprocessing, architectures, splits, and seed aggregation; verification of the frozen numerical record does not require retraining.

