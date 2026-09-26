# Revision uncertainty analyses

Regenerate all three tables with:

```bash
python scripts/bootstrap_revision_uncertainty.py
```

- `xai_patient_clustered_bootstrap.csv`: 10,000-replicate percentile intervals for pointing game, saliency mass, and IoU@0.5. Patients are resampled with replacement and all their slices are retained.
- `patient_accuracy_difference_bootstrap.csv`: paired patient-level accuracy differences (BrNet minus baseline) with 10,000-replicate percentile intervals.
- `phash_threshold_sensitivity.csv`: match-pair and unique-external-image counts at 256-bit pHash distances 0, 5, and 10.

The bootstrap seed is 20260925. The canonical corrected MobileNetV2 table is used alongside the archived EfficientNetB7 and ViT-B/16 outputs. The script checks the canonical patient accuracies and pHash counts before writing its outputs.
