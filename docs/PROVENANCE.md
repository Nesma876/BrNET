# Artifact provenance and evidence status

## Fully supported archived results

- Four-model patient-disjoint OOF accuracies.
- Patient-level aggregation for 233 patients.
- Exact two-sided McNemar tests and Holm correction at slice and patient level.
- BrNet ECE and multiclass Brier score from pooled OOF probabilities.
- pHash totals and denominator distinctions.
- Grad-CAM localization summaries directly aggregated from the surviving row-level table.
- Executable BrNet architecture dimensions and parameter counts.

## Later verified outputs with partial generating provenance

### External predictive performance

A later 452-row prediction table records 233 correct predictions after median filtering (51.55%; macro-F1 0.4235). The probability columns reproduce every recorded class prediction, and the 1,137-row exclusion manifest reconciles to 685 excluded and 452 retained images. The exact executed script and checkpoint file were not delivered with this table; only the checkpoint SHA-256 survives in the execution log. This is reported as a verified retrospective screened-subset output, not as independent external validation.

### Matched-area perturbation

A later 2,594-row perturbation table contains 86 unavailable matched controls and 2,508 analyzable pairs. Direct patient-clustered bootstrap recomputation reproduces the reported estimates and intervals. The exact control-region coordinates or masks, the exact executed script, and hashes for the 15 fold/seed checkpoints were not delivered. The numerical output is therefore verified, but its full generating provenance is incomplete.

### Grad-CAM target aggregation

The row-level localization table is retained. The seed-level checkpoints and maps required to independently re-establish whether all seeds used one pooled target class remain absent. The repository does not claim that procedure as independently reconstructed.

### Parameter randomization

A later archive contains a 100-image sample manifest and 600 image-step similarity records for one checkpoint. Valid Spearman correlations range from 50/100 at the first step to 92--95/100 later because some maps are constant. Repeated GPU runs were not bit-identical. The analysis is descriptive only and no formal pass/fail claim is made.

### Compute benchmark

A later archive contains 100 batch-1 latency measurements and 20 batch-32 throughput measurements for each of BrNet, MobileNetV2, and EfficientNetB7, plus an environment log. BrNet had the lowest recorded median latency and highest recorded throughput in that run. ViT-B/16 was not measured, concurrent GPU activity was not recorded, and one BrNet latency measurement was a large outlier. Hardware-independent runtime superiority is not claimed.

Machine-readable files, hashes, and detailed limitations are in `results/audit_v2/`.

## pHash denominator rules

- 685: unique external images with at least one detected overlap.
- 880: individual external–benchmark match pairs.
- 466: exact-distance-zero match pairs.
- 433: unique external images represented by those 466 exact pairs.
- 561/78/79: unique external images matching training/validation/test, respectively; categories may overlap across partitions.
- 81.9%: 561/685, not 561/880.
