# Frozen model comparison

Generated from four aligned canonical OOF tables: 3064 images, 233 patients.

| analysis_level | model_or_comparison | n | correct | accuracy | n00 | n01 | n10 | n11 | exact_p | holm_adjusted_p | reject_0_05 | interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| slice | BrNet | 3064 | 2594 | 0.8466057441 |  |  |  |  |  |  |  | descriptive/secondary |
| patient | BrNet | 233 | 201 | 0.8626609442 |  |  |  |  |  |  |  | primary inferential unit |
| slice | MobileNetV2 corrected | 3064 | 2692 | 0.8785900783 |  |  |  |  |  |  |  | descriptive/secondary |
| patient | MobileNetV2 corrected | 233 | 204 | 0.8755364807 |  |  |  |  |  |  |  | primary inferential unit |
| slice | EfficientNetB7 | 3064 | 2728 | 0.8903394256 |  |  |  |  |  |  |  | descriptive/secondary |
| patient | EfficientNetB7 | 233 | 209 | 0.8969957082 |  |  |  |  |  |  |  | primary inferential unit |
| slice | ViT-B/16 | 3064 | 2654 | 0.8661879896 |  |  |  |  |  |  |  | descriptive/secondary |
| patient | ViT-B/16 | 233 | 210 | 0.9012875536 |  |  |  |  |  |  |  | primary inferential unit |
| slice | BrNet vs MobileNetV2 corrected | 3064 |  |  | 185 | 285 | 187 | 2407 | 7.466226352e-06 | 1.49324527e-05 | True | descriptive/secondary |
| slice | BrNet vs EfficientNetB7 | 3064 |  |  | 154 | 316 | 182 | 2412 | 2.032110204e-09 | 6.096330613e-09 | True | descriptive/secondary |
| slice | BrNet vs ViT-B/16 | 3064 |  |  | 196 | 274 | 214 | 2380 | 0.007503832239 | 0.007503832239 | True | descriptive/secondary |
| patient | BrNet vs MobileNetV2 corrected | 233 |  |  | 16 | 16 | 13 | 188 | 0.7110711038 | 0.7110711038 | False | primary paired inference; nonsignificance is not equivalence |
| patient | BrNet vs EfficientNetB7 | 233 |  |  | 11 | 21 | 13 | 188 | 0.229481013 | 0.4589620261 | False | primary paired inference; nonsignificance is not equivalence |
| patient | BrNet vs ViT-B/16 | 233 |  |  | 14 | 18 | 9 | 192 | 0.1220781207 | 0.3662343621 | False | primary paired inference; nonsignificance is not equivalence |

Slice-level comparisons are secondary/descriptive because slices are clustered within patients. Patient-level comparisons are the primary inferential analysis. A non-significant patient-level result is not evidence of equivalence or non-inferiority.

## Canonical hashes

- `oof_predictions_full.csv`: `1a08c79faa7dd6bacf70eea0103802f9aa2866961d9ccd4b13d0a96cbd3fa1e2`
- `oof_predictions_mobilenetv2_v2.csv`: `57229b8af5a5e17f199796fe455de7a699434fb844b458da64b8e0d69f219c0f`
- `oof_predictions_baselines.csv`: `7261797cacc1bd4d79807a8664e50c75b19aa063f3e3346499429d00dd671ee5`
- `oof_predictions_vit.csv`: `5191c29ace3e2487d755687a5f0419e91e444db57003e2ae146faf72d10846cb`
