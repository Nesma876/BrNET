# Provenance and evidential scope

This public package contains only the evidence used by the current submission.

## Included

- Canonical patient-disjoint OOF predictions for BrNet and the three frozen-backbone baselines.
- One fixed fold assignment per patient and lightweight dataset manifests.
- Exact patient-level paired inference and paired patient-bootstrap accuracy-difference intervals.
- Complete pHash match-pair output with unique-image and partition summaries.
- Archived row-level Grad-CAM localization metrics and patient-clustered confidence intervals.
- Configuration, analysis, verification, and reference experiment code.

## Not included or not claimed

- Raw MRI images, because they remain with their public data providers.
- Model checkpoints.
- Manuscript or response-letter source files.
- Superseded results, intermediate audit reports, exploratory runtime measurements, and incomplete historical artifacts.
- Independent reconstruction of the historical seed-level Grad-CAM target-selection process; localization claims are limited to the archived row-level metrics.
- Independent external validation: the nominal external source overlaps substantially with the development benchmark and is used only for provenance auditing.

## pHash denominators

- 685: unique external images with at least one match at $d\leq10$.
- 880: individual external--benchmark match pairs at $d\leq10$.
- 433/466: unique images/match pairs at $d=0$.
- 681/813: unique images/match pairs at $d\leq5$.
- 561/78/79: unique images matching training/validation/test; these partition categories are not mutually exclusive.

