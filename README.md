# BrNet reproducibility repository

[![Verify archived results](https://github.com/Nesma876/BrNET/actions/workflows/verify.yml/badge.svg)](https://github.com/Nesma876/BrNET/actions/workflows/verify.yml)

Code and frozen numerical outputs accompanying the BrNet major revision.

The repository separates three evidence levels:

1. **Frozen and reproducible from archived prediction tables**: patient-disjoint OOF metrics, patient aggregation, exact McNemar tests, Holm correction, calibration, pHash denominator accounting, architecture dimensions, and row-level Grad-CAM localization summaries.
2. **Verified from newly archived row-level outputs, with partial generating provenance**: screened external predictions and matched-area perturbation outputs. Their numerical summaries can be recalculated, but the exact executed scripts and required checkpoint files were not included with the received archive.
3. **Descriptive or environment-specific audits**: the single-checkpoint Grad-CAM randomization analysis and the three-CNN compute benchmark. These do not establish a formal XAI pass or hardware-independent runtime superiority.

## Canonical patient-disjoint results

| Model | Slice accuracy | Slice macro-F1 | Slice macro-AUC | Patient accuracy |
|---|---:|---:|---:|---:|
| BrNet | 84.6606% | 82.9441% | 95.4447% | 86.2661% |
| MobileNetV2 (corrected preprocessing) | 87.8590% | 86.4618% | 96.3524% | 87.5536% |
| EfficientNetB7 | 89.0339% | 87.3442% | 97.0478% | 89.6996% |
| ViT-B/16 | 86.6188% | 85.0866% | 96.3137% | 90.1288% |

The patient-disjoint Figshare evaluation contains three tumor classes only (glioma, meningioma, and pituitary tumor); it does not evaluate patient-level healthy-versus-tumor discrimination.

Canonical document-control values:

- BrNet pooled OOF: **2,594/3,064 = 84.6606%** at slice level and **201/233 = 86.2661%** at patient level.
- Patient-level exact/Holm p-values: MobileNetV2 **0.7110711/0.7110711**; EfficientNetB7 **0.2294810/0.4589620**; ViT-B/16 **0.1220781/0.3662344**.
- Paired patient-level accuracy differences (BrNet minus baseline; 95% patient-bootstrap CI): MobileNetV2 **−1.29 pp [−6.01, 3.00]**; EfficientNetB7 **−3.43 pp [−8.15, 1.29]**; ViT-B/16 **−3.86 pp [−8.15, 0.43]**.
- Provenance audit: **685/1,137** unique external images with a match; **880** match pairs; **466** exact-distance-zero pairs involving **433** unique external images; train/validation/test unique-image counts **561/78/79**.
- pHash sensitivity: **433/466** unique images/pairs at distance 0, **681/813** at distance ≤5, and **685/880** at distance ≤10.
- Figure 3, in display order: benchmark/external **0375.jpg/186.jpg**, **0799.jpg/154.jpg**, **0074.jpg/130.jpg**, and **1501.jpg/192.jpg**; all four have pHash distance zero.
- The historical **70.84%** value is withdrawn as unreconstructable; it is not interpreted as directly comparable to or corrected by **84.6606%**.
- The historical **96.44%** value is retained only as a descriptive benchmark and is not used for model selection, uncertainty estimation, or inferential comparison.

Additional audited outputs received after the initial artifact freeze are documented in [`results/audit_v2/`](results/audit_v2/README.md):

- screened nominally external subset: **233/452 = 51.55%** accuracy and macro-F1 **0.4235**; this is a retrospective result on a highly imbalanced residual subset, not independent external validation;
- matched-area perturbation: **2,594** eligible images, **86** unavailable controls, and **2,508** analyzable pairs; the patient-clustered paired differences are **0.0578** for class-change rate (95% CI **0.0398–0.0790**) and **0.0371** for target-class probability (95% CI **0.0249–0.0520**);
- Grad-CAM parameter randomization: exploratory single-checkpoint results only; mean $\rho$ was **0.5688** for 50/100 valid correlations at step 1 and **0.1198** for 95/100 at step 2, with **92–95/100** valid correlations thereafter;
- compute benchmark: BrNet median batch-1 latency **74.70 ms** and batch-32 throughput **333.89 images/s**, compared descriptively with MobileNetV2 and EfficientNetB7 under the recorded environment.

Slice-level paired comparisons are secondary/descriptive because slices are clustered within patients. Patient-level comparisons are the primary inferential analysis. Nonsignificant patient-level results are not interpreted as equivalence or non-inferiority.

The exact frozen tables are in [`results/FROZEN_MODEL_COMPARISON.csv`](results/FROZEN_MODEL_COMPARISON.csv). The published row-level OOF tables can be checked in one command:

```bash
python scripts/verify_frozen_results.py
```

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the checks performed.

## Repository layout

- `scripts/`: reference experiment and revision scripts preserving the documented preprocessing and model definitions; later received outputs may not have byte-identical generating scripts.
- `src/`: reusable data, model, evaluation, statistics, and XAI modules.
- `configs/`: frozen protocol and model configurations.
- `data/manifests/`: lightweight dataset manifests; raw MRI data are not distributed.
- `results/`: canonical small numerical tables used in the paper revision.
- `results/audit_v2/`: later row-level outputs, raw compute timings, execution logs, and an evidence-level manifest.
- `results/revision_uncertainty/`: patient-clustered Grad-CAM intervals, paired patient-level accuracy-difference intervals, and pHash threshold sensitivity.
- `tests/`: integrity and metric tests.
- `docs/`: provenance, script inventory, and known limitations.
- `supplementary/`: submission-ready `Online_Resource_1.tex`, fold-by-seed stability tables, and the machine-readable Figure 3 selection.

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
- The screened external prediction table is archived, but the four-class checkpoint file and exact executed script are not. Its recorded checkpoint SHA-256 is provided in the execution log.
- The matched-area row-level output is archived, but the 15 checkpoint hashes and exact control-region coordinates/masks are not.
- The Grad-CAM randomization analysis uses one checkpoint and is not bit-identical across repeated GPU runs; it is descriptive only.
- Raw compute timings and an environment log are archived for three CNNs. ViT-B/16 was not measured, concurrent GPU activity was not recorded, and one BrNet latency run was an outlier. Runtime results are environment-specific and descriptive.

## Key implementation details

- BrNet patient-disjoint OOF: five patient-disjoint folds, seeds 42/43/44, pooled by averaging class probabilities before argmax.
- MobileNetV2: dedicated `mobilenet_v2.preprocess_input` correction.
- ViT-B/16: `timm` `vit_base_patch16_224`, frozen backbone, trainable `768 → 128 → 3` readout containing **98,819 trainable parameters**. The value 98,948 applies to a four-class head and not to the three-class patient-disjoint experiment. Exact preprocessing and arithmetic are documented in [`docs/VIT_PREPROCESSING.md`](docs/VIT_PREPROCESSING.md).
- pHash: 16×16 perceptual hashes; distance 0 is the strongest duplication evidence and distance ≤10 is an inclusive screening threshold. Sensitivity at distances 0, 5, and 10 is archived; unique-image counts and match-pair counts are reported separately.
- Fold-by-seed reporting: all 15 BrNet, MobileNetV2, and EfficientNetB7 runs are archived in `supplementary/`. Only six ViT-B/16 intermediate runs survived, so the ViT table is explicitly labeled partial.
- Figure 3: its four exact-distance-zero examples are tied to `supplementary/figure3_selected_pairs.csv`; quantitative conclusions use the complete 880-pair table.

## Citation and license

Please cite the associated paper when using this code. Machine-readable citation metadata, including the complete author list and affiliations, are provided in [`CITATION.cff`](CITATION.cff).

A software license has not been added because no license choice was supplied by the authors; repository users should therefore treat the code as all rights reserved until a license is selected.
