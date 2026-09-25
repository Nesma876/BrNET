# Supplementary reproducibility tables

- `Online_Resource_1.tex`: submission-ready supplementary document containing Table S1, pHash threshold sensitivity, and explicitly labeled secondary matched-area, parameter-randomization, and compute tables.

- `fold_seed_brnet.csv`: all 15 archived BrNet runs (5 folds × 3 seeds). `n_patients_test` is available; train/validation patient counts were not present in the surviving summary and are not reconstructed here.
- `fold_seed_mobilenetv2_efficientnetb7.csv`: all 15 archived runs for each of MobileNetV2 and EfficientNetB7.
- `fold_seed_vit_partial.csv`: the six surviving ViT-B/16 intermediate runs (folds 4 and 5 only). This is explicitly incomplete and must not be presented as a full fold-by-seed summary.
- `figure3_selected_pairs.csv`: the four exact-distance-zero pairs displayed in Figure 3. They are the first four distinct external images after filtering the frozen complete pair table to distance zero and preserving its archived row order.
- `figures/pipeline.png`: simplified Figure 1 with enlarged internal text.
- `figures/phash_verification.png`: Figure 3 with enlarged labels; MRI pixel content is taken unchanged from `phash_verification_source.png` and can be re-laid out with `scripts/regenerate_submission_figures.py`.

Run-level accuracies are stability diagnostics. The primary pooled OOF accuracy is computed from one probability-averaged prediction per held-out image and is not the arithmetic mean of the run-level accuracies.
