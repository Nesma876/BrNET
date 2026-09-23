"""
BrNet -- Régénération de la figure Grad-CAM (cohérence avec Table XAI finale)
================================================================================
Régénère gradcam_sanity_check_compact.png en utilisant les MÊMES modèles que
ceux ayant produit tous les chiffres de la Table XAI finale et de la section
masking (les 15 poids brnet_fold{F}_seed{S}_oof.weights.h5, moyennés par
fold). L'ancienne figure datait d'avant cette révision et utilisait un
modèle différent -- incohérence signalée et corrigée ici.

Nécessite : les 15 fichiers brnet_fold{F}_seed{S}_oof.weights.h5 et le
dossier Figshare regroupé (figshare_raw), dans la même session.
"""

import os
import numpy as np
import pandas as pd
import h5py
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

FIGSHARE_DIR = "/kaggle/working/figshare_raw"
IMG_SIZE = 256
SEEDS = [42, 43, 44]
N_FOLDS = 5
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
RNG_SEED = 123


def load_figshare_mat_files(mat_dir):
    images_raw, labels, pids, masks = [], [], [], []
    mat_files = sorted(f for f in os.listdir(mat_dir) if f.endswith(".mat"))
    for fname in mat_files:
        path = os.path.join(mat_dir, fname)
        with h5py.File(path, "r") as f:
            cjdata = f["cjdata"]
            img = np.array(cjdata["image"]).astype(np.float32)
            label = int(np.array(cjdata["label"]).squeeze())
            pid = np.array(cjdata["PID"])
            pid = "".join(chr(c) for c in pid.flatten()) if pid.dtype.kind in "OU" else str(pid)
            mask = np.array(cjdata["tumorMask"]).astype(np.uint8)

        img_norm01 = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_uint8 = (img_norm01 * 255).astype(np.uint8)
        img_resized = cv2.resize(img_uint8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        images_raw.append(img_resized)
        labels.append(label - 1)
        pids.append(pid)
        masks.append(mask_resized)

    return np.stack(images_raw), np.array(labels), np.array(pids), np.stack(masks)


def finalize_image(img_uint8):
    filtered = cv2.medianBlur(img_uint8, 3)
    return filtered.astype(np.float32)[..., None]


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
])


def build_brnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=3):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", name="last_conv")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def grad_cam(model, img, layer_name="last_conv"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img[None, ...])
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(weights * conv_out[0], axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam, int(class_idx.numpy())


def pointing_game_hit(cam, mask):
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return None
    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    return int(mask_bin[py, px] == 1)


if __name__ == "__main__":
    print("=== Chargement Figshare ===")
    X_raw, y, pids, masks = load_figshare_mat_files(FIGSHARE_DIR)
    X = np.stack([finalize_image(img) for img in X_raw])
    n_total = len(X)
    print(f"{n_total} images, {len(set(pids))} patients")

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(X, y, pids))
    fold_assignment = np.full(n_total, -1)
    for fold_idx, (_, test_idx) in enumerate(splits):
        fold_assignment[test_idx] = fold_idx

    has_mask = masks.sum(axis=(1, 2)) > 0

    print("\n=== Calcul Grad-CAM (per-fold, 3-seed average) sur les images avec masque ===")
    cams_all = {}
    pointing_hits = {}
    preds_all = np.full(n_total, -1)

    for fold_idx in range(N_FOLDS):
        idx_fold = np.where((fold_assignment == fold_idx) & has_mask)[0]
        print(f"Fold {fold_idx+1}: {len(idx_fold)} images")

        cams_per_seed = {seed: {} for seed in SEEDS}
        preds_per_seed = {seed: {} for seed in SEEDS}
        for seed in SEEDS:
            weights_path = f"brnet_fold{fold_idx+1}_seed{seed}_oof.weights.h5"
            model = build_brnet()
            model.load_weights(weights_path)
            for i in idx_fold:
                cam, pred = grad_cam(model, X[i], "last_conv")
                cams_per_seed[seed][i] = cam
                preds_per_seed[seed][i] = pred
            del model
            tf.keras.backend.clear_session()

        for i in idx_fold:
            cam_avg = np.mean([cams_per_seed[s][i] for s in SEEDS], axis=0)
            cams_all[i] = cam_avg
            hit = pointing_game_hit(cam_avg, masks[i])
            pointing_hits[i] = hit
            seed_preds = [preds_per_seed[s][i] for s in SEEDS]
            preds_all[i] = max(set(seed_preds), key=seed_preds.count)

    print("\n=== Sélection des exemples représentatifs ===")
    rng = np.random.RandomState(RNG_SEED)
    selected = {}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        candidates = [i for i in cams_all if y[i] == class_idx and preds_all[i] == y[i]]
        if not candidates:
            continue
        chosen = rng.choice(candidates)
        selected[class_name] = chosen
        print(f"{class_name}: image_idx={chosen}, pointing_hit={pointing_hits[chosen]}")

    print("\n=== Génération de la figure ===")
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    if len(selected) == 1:
        axes = axes[None, :]

    for row, (class_name, idx) in enumerate(selected.items()):
        img = X_raw[idx]
        mask = masks[idx]
        cam = cams_all[idx]

        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(f"Image ({class_name})")
        axes[row, 0].axis("off")

        overlay = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
        mask_color = np.zeros_like(overlay)
        mask_color[..., 0] = mask > 0
        overlay_masked = overlay * 0.7 + mask_color * 0.3
        axes[row, 1].imshow(overlay_masked)
        axes[row, 1].set_title("Masque réel (rouge)")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].imshow(cam, cmap="jet", alpha=0.5)
        pred_class = CLASS_NAMES[preds_all[idx]]
        axes[row, 2].set_title(f"Grad-CAM (pred={pred_class})")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_sanity_check_v2.png", dpi=150, bbox_inches="tight")
    print("\nFigure sauvegardée : gradcam_sanity_check_v2.png")
