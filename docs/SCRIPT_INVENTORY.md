# Script inventory

| Script | Purpose | Evidence status |
|---|---|---|
| `patient_disjoint_rigorous_oof.py` | BrNet five-fold, three-seed patient-disjoint OOF | Canonical method; requires raw data and checkpoints/training |
| `baselines_patient_disjoint_oof.py` | Corrected MobileNetV2 and EfficientNetB7 OOF | Canonical method |
| `vit_patient_disjoint_oof.py` | Frozen-backbone ViT-B/16 OOF | Canonical method |
| `point2_patient_level_aggregation.py` | BrNet patient aggregation | Reproducible from OOF CSV |
| `mcnemar_patient_level.py` | Four-model patient aggregation and McNemar inputs | Reproducible from OOF CSVs |
| `calibration_analysis.py` | ECE, Brier and classwise diagnostic metrics | Reproducible from BrNet OOF CSV |
| `point4_phash_duplicates.py` | Cross-source pHash pair generation | Requires source images |
| `phash_partition_breakdown.py` | Partition-specific overlap accounting | Requires historical split reconstruction |
| `phash_robustness_check.py` | Threshold and class sensitivity | Requires pHash pairs/source inventory |
| `point6_layer_dimensions.py` | BrNet geometry and parameter count | Reproducible |
| `point5_xai_masking_v2_fixed.py` | Fixed-target Grad-CAM and fixed matched control | Code retained; historical corrected output not independently reconstructable |
| `regenerate_gradcam_figure.py` | Grad-CAM figure generation | Requires 15 checkpoints |
| `xai_sanity_check.py` | Cascading parameter randomization | Requires checkpoint and images |
| `external_validation_clean.py` | Corrected median-filter external inference | Requires absent four-class checkpoint |
| `compute_benchmark.py` | Historical timing implementation | Exploratory; raw historical timing logs absent |
| `ablation_study.py` | Legacy image-level ablation | Historical benchmark analysis |
| `mcnemar_test.py` | Legacy image-level paired comparison | Historical benchmark analysis |
| `mobilenetv2_benchmark_fixed.py` | Corrected image-level MobileNetV2 preprocessing | Historical benchmark correction |
