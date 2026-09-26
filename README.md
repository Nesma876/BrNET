# BrNet reviewer reproducibility package

This repository contains the code, frozen row-level outputs, numerical summaries, and lightweight manifests needed to inspect the current journal submission. It intentionally excludes manuscript sources, raw MRI data, checkpoints, superseded analyses, and historical audit archives.

## Verify the reported results

Python 3.12 is recommended. GPU access is not required for verification from the frozen outputs.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_frozen_results.py
python scripts/bootstrap_revision_uncertainty.py
```

The verification script checks alignment across 3,064 slices and 233 patients, the four slice- and patient-level accuracies, discordant-pair counts, exact McNemar tests, and Holm-adjusted p-values. The bootstrap script regenerates patient-level accuracy-difference intervals, patient-clustered Grad-CAM intervals, and pHash threshold sensitivity.

## Current numerical record

| Model | Slice accuracy | Slice macro-F1 | Slice macro-AUC | Patient accuracy |
|---|---:|---:|---:|---:|
| BrNet | 84.6606% | 82.9441% | 95.4447% | 86.2661% |
| MobileNetV2 | 87.8590% | 86.4618% | 96.3524% | 87.5536% |
| EfficientNetB7 | 89.0339% | 87.3442% | 97.0478% | 89.6996% |
| ViT-B/16 | 86.6188% | 85.0866% | 96.3137% | 90.1288% |

The primary paired inference is performed after aggregation to one prediction per patient. The Holm-adjusted p-values for BrNet versus MobileNetV2, EfficientNetB7, and ViT-B/16 are 0.7110711, 0.4589620, and 0.3662344. Nonsignificance is not interpreted as equivalence or non-inferiority.

The provenance audit identifies 685 of 1,137 external images with at least one benchmark match, represented by 880 match pairs at pHash distance $d\leq10$. Threshold sensitivity is 433/466 unique images/pairs at $d=0$, 681/813 at $d\leq5$, and 685/880 at $d\leq10$.

## Repository layout

- `results/row_level/`: canonical OOF predictions, pHash pairs, and Grad-CAM localization rows.
- `results/revision_uncertainty/`: current bootstrap intervals and pHash sensitivity.
- `results/*.csv`: compact tables reported in the revision.
- `scripts/`: verification, analysis, and reference experiment implementations.
- `src/`: reusable evaluation, statistics, model, and XAI code.
- `configs/`: frozen model and protocol configurations.
- `data/manifests/`: lightweight public-dataset and patient-fold manifests; no images are distributed.
- `supplementary/`: machine-readable tables supporting the online resource and figure selection.
- `docs/`: provenance limits and protocol details.

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the verification map and [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the precise evidential scope.

## Data and citation

Raw medical images and model checkpoints are not distributed. Public dataset identifiers and file manifests are provided under `data/manifests/`. Please cite the associated paper; machine-readable metadata are available in [`CITATION.cff`](CITATION.cff).
