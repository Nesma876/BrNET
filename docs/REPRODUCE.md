# Reproducing the archived numerical results

## Frozen model comparison

The four canonical row-level OOF tables are published under `results/row_level/`. Run:

```bash
python scripts/verify_frozen_results.py
```

The command independently checks:

- exact alignment of image, patient, fold, and reference-label fields;
- 3,064 slices and 233 patients;
- one test fold and one reference class per patient;
- all four slice accuracies;
- majority-vote patient aggregation with probability tie-breaking;
- all four patient accuracies;
- all slice- and patient-level McNemar cells;
- exact two-sided p-values and Holm-adjusted p-values;
- equality with `results/FROZEN_MODEL_COMPARISON.csv`.

## Grad-CAM localization

`results/row_level/results_xai_full_stratified_v2.csv` contains one row per image. The publication table `results/final_xai_metrics.csv` is obtained by grouping on `correct` and `class` and taking the mean of the localization columns. This row-level artifact supports descriptive localization results, but not independent reconstruction of the historical seed-level target procedure because the checkpoints and maps were unavailable.

## pHash overlap

`results/row_level/results_phash_duplicates.csv` contains the 880 external–benchmark match pairs. Unique external-image counts must be computed with `external_path.nunique()`. Exact matches are rows with `hamming_distance == 0`. The partition summary is retained separately because the pair table does not contain a partition field.

## Full experiment scripts

Training and image-level inference scripts preserve their historical Kaggle paths as provenance. Before running them elsewhere, edit the dataset/checkpoint constants at the top of each script. Do not change preprocessing, folds, seeds, or aggregation rules when attempting reproduction.
