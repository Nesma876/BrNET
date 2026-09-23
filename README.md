# BrNet reproducibility repository

[![Verify archived results](https://github.com/Nesma876/BrNET/actions/workflows/verify.yml/badge.svg)](https://github.com/Nesma876/BrNET/actions/workflows/verify.yml)

Code and frozen numerical outputs accompanying the BrNet major revision.

The repository separates two evidence levels:

1. **Frozen and reproducible from archived prediction tables**: patient-disjoint OOF metrics, patient aggregation, exact McNemar tests, Holm correction, calibration, pHash denominator accounting, architecture dimensions, and row-level Grad-CAM localization summaries.
2. **Code available, artifacts incomplete**: corrected external inference, fixed matched-area perturbation, parameter-randomization reconstruction, and historical GPU timing. These scripts are included for transparency, but their historical numerical summaries must not be treated as independently reproduced without the checkpoints or raw outputs listed in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## Canonical patient-disjoint results

| Model | Slice accuracy | Patient accuracy |
|---|---:|---:|
| BrNet | 84.6606% | 86.2661% |
| MobileNetV2 (corrected preprocessing) | 87.8590% | 87.5536% |
| EfficientNetB7 | 89.0339% | 89.6996% |
| ViT-B/16 | 86.6188% | 90.1288% |

Slice-level paired comparisons are secondary/descriptive because slices are clustered within patients. Patient-level comparisons are the primary inferential analysis. Nonsignificant patient-level results are not interpreted as equivalence or non-inferiority.

The exact frozen tables are in [`results/FROZEN_MODEL_COMPARISON.csv`](results/FROZEN_MODEL_COMPARISON.csv). The published row-level OOF tables can be checked in one command:

```bash
python scripts/verify_frozen_results.py
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the checks performed.

## Repository layout

- `scripts/`: historical experiment and revision scripts, preserved with their exact preprocessing and model definitions.
- `src/`: reusable data, model, evaluation, statistics, and XAI modules.
- `configs/`: frozen protocol and model configurations.
- `data/manifests/`: lightweight dataset manifests; raw MRI data are not distributed.
- `results/`: canonical small numerical tables used in the paper revision.
- `tests/`: integrity and metric tests.
- `docs/`: provenance, script inventory, and known limitations.

## Environment

Core dependencies are pinned in [`requirements.txt`](requirements.txt). Historical experiment scripts additionally use the packages listed in [`requirements-experiments.txt`](requirements-experiments.txt); their exact historical versions were not archived, which is stated explicitly rather than guessed. GPU training is not required to recompute the frozen metrics from the published prediction CSVs.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_frozen_results.py
```

Dataset and checkpoint locations in the historical Kaggle scripts are constants near the top of each file. Adapt them to the local environment without changing preprocessing or aggregation rules.

## Important reproducibility limits

- Raw medical images are not committed.
- Model checkpoints are not committed.
- The corrected four-class checkpoint needed for the contamination-screened external evaluation was unavailable at final audit; no corrected external accuracy is claimed here.
- The 15 BrNet fold/seed checkpoints needed to reconstruct corrected matched-area and seed-level Grad-CAM procedures were unavailable at final audit.
- Historical runtime summaries lacked raw timing logs and are not included as evidence of controlled speed superiority.

## Key implementation details

- BrNet patient-disjoint OOF: five patient-disjoint folds, seeds 42/43/44, pooled by averaging class probabilities before argmax.
- MobileNetV2: dedicated `mobilenet_v2.preprocess_input` correction.
- ViT-B/16: `timm` `vit_base_patch16_224`, frozen backbone, trainable `768 → 128 → 3` readout containing **98,819 trainable parameters**. The value 98,948 applies to a four-class head and not to the three-class patient-disjoint experiment. Exact preprocessing and arithmetic are documented in [`docs/VIT_PREPROCESSING.md`](docs/VIT_PREPROCESSING.md).
- pHash: 16×16 perceptual hashes with Hamming-distance threshold 10/256; unique-image counts and match-pair counts are reported separately.

## Citation and license

Please cite the associated paper when using this code. A software license has not been added because no license choice was supplied by the authors; repository users should therefore treat the code as all rights reserved until a license is selected.
