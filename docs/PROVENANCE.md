# Artifact provenance and evidence status

## Fully supported archived results

- Four-model patient-disjoint OOF accuracies.
- Patient-level aggregation for 233 patients.
- Exact two-sided McNemar tests and Holm correction at slice and patient level.
- BrNet ECE and multiclass Brier score from pooled OOF probabilities.
- pHash totals and denominator distinctions.
- Grad-CAM localization summaries directly aggregated from the surviving row-level table.
- Executable BrNet architecture dimensions and parameter counts.

## Results not claimed as independently reproduced

### External predictive performance

The historical detailed external CSV belongs to a superseded preprocessing pipeline. The corrected four-class checkpoint required to regenerate a contamination-screened row-level evaluation was absent. The repository therefore retains the provenance/overlap audit but does not claim a corrected clean-subset accuracy.

### Matched-area perturbation

The corrected script fixes the control region once per image and reuses it across seeds, but the 15 fold/seed checkpoints and corrected row-level output were absent at final audit. Historical summary values are not included as finalized evidence.

### Grad-CAM target aggregation

The row-level localization table is retained. The seed-level checkpoints and maps required to independently re-establish whether all seeds used one pooled target class were absent. The repository does not claim that procedure as independently reproduced.

### Parameter randomization

Only an aggregate summary survived; per-image maps and complete generating metadata were unavailable. No formal pass/fail claim is made.

### Compute benchmark

The historical summary lacked raw timings and sufficient protocol metadata. Runtime superiority is not claimed.

## pHash denominator rules

- 685: unique external images with at least one detected overlap.
- 880: individual external–benchmark match pairs.
- 466: exact-distance-zero match pairs.
- 433: unique external images represented by those 466 exact pairs.
- 561/78/79: unique external images matching training/validation/test, respectively; categories may overlap across partitions.
- 81.9%: 561/685, not 561/880.
