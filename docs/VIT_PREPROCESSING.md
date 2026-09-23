# ViT-B/16 preprocessing and readout

Source: `scripts/vit_patient_disjoint_oof.py`.

Each Figshare MRI slice is processed as follows:

1. Per-image min–max normalization.
2. Conversion to unsigned 8-bit intensity.
3. 3×3 median filtering.
4. Resize to 224×224 with OpenCV `INTER_AREA` interpolation.
5. Division by 255 to obtain `[0,1]` intensities.
6. Horizontal flip with probability 0.5 during training only.
7. Replication of the grayscale image into three identical channels.
8. Channel-wise ImageNet normalization with mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
9. Conversion from HWC to CHW layout.

The model is `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)`. All backbone parameters are frozen. The trainable readout is:

```text
Linear(vit.num_features, 128)
ReLU
Dropout(0.3)
Linear(128, 3)
```

Only the readout is optimized, using Adam with learning rate `2e-5`. Cross-entropy operates on logits. At inference, softmax probabilities are computed independently for seeds 42, 43, and 44, averaged for each held-out slice, and converted to the final prediction by argmax.

SHA-256 of the archived source script: `53239bd373b9b85dca0e46712b8abc409d5ff9fa81c383c163c566c0d5439d09`.
