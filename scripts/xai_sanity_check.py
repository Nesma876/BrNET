"""
BrNet -- XAI Sanity Check : Cascading Parameter Randomization (Point 14)
================================================================================
Test de robustesse standard pour les méthodes d'explicabilité (Adebayo et al.
2018, "Sanity Checks for Saliency Maps") : si Grad-CAM ne change PAS quand on
randomise les poids du modèle, ça prouve que la carte de saillance ne dépend
pas de ce que le modèle a appris.

Protocole : randomisation CASCADE, couche par couche, de la dernière vers la
première. Pour chaque étape, on recalcule Grad-CAM sur un échantillon et on
mesure la similarité (Spearman, SSIM approx) avec la carte originale.

Nécessite : un modèle BrNet déjà entraîné (ex. brnet_fold1_seed42_oof.weights.h5)
et Figshare (images + masques). Calcul rapide (échantillon ~100 images).
"""

import os
import numpy as np
import pandas as pd
import h5py
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.stats import spearmanr

FIGSHARE_DIR = "/kaggle/working/figshare_raw"
IMG_SIZE = 256
N_CLASSES = 3
N_SAMPLE_IMAGES = 100
RNG_SEED = 42
WEIGHTS_PATH = "brnet_fold1_seed42_oof.weights.h5"

LAYER_ORDER_CASCADE = ["dense_out", "dense1", "last_conv", "conv3", "conv2", "conv1"]


def load_figshare_mat_files(mat_dir):
    images_raw, labels, masks = [], [], []
    mat_files = sorted(f for f in os.listdir(mat_dir) if f.endswith(".mat"))
    for fname in mat_files:
        path = os.path.join(mat_dir, fname)
        with h5py.File(path, "r") as f:
            cjdata = f["cjdata"]
            img = np.array(cjdata["image"]).astype(np.float32)
            label = int(np.array(cjdata["label"]).squeeze())
            mask = np.array(cjdata["tumorMask"]).astype(np.uint8)

        img_norm01 = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_uint8 = (img_norm01 * 255).astype(np.uint8)
        img_resized = cv2.resize(img_uint8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        images_raw.append(img_resized)
        labels.append(label - 1)
        masks.append(mask_resized)

    return np.stack(images_raw), np.array(labels), np.stack(masks)


def finalize_image(img_uint8):
    filtered = cv2.medianBlur(img_uint8, 3)
    return filtered.astype(np.float32)[..., None]


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
])


def build_brnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=N_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", name="conv1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", name="last_conv")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="dense_out")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def randomize_layer_weights(model, layer_name, rng):
    layer = model.get_layer(layer_name)
    weights = layer.get_weights()
    new_weights = []
    for w in weights:
        std = w.std() if w.std() > 0 else 0.01
        new_w = rng.normal(loc=0.0, scale=std, size=w.shape).astype(w.dtype)
        new_weights.append(new_w)
    layer.set_weights(new_weights)


def grad_cam_fixed_class(model, img, target_class, layer_name="last_conv"):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img[None, ...])
        loss = preds[:, target_class]
    grads = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(weights * conv_out[0], axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam


def cam_similarity(cam_a, cam_b):
    flat_a = cam_a.flatten()
    flat_b = cam_b.flatten()
    rho, _ = spearmanr(flat_a, flat_b)

    mu_a, mu_b = flat_a.mean(), flat_b.mean()
    var_a, var_b = flat_a.var(), flat_b.var()
    cov_ab = np.mean((flat_a - mu_a) * (flat_b - mu_b))
    c1, c2 = (0.01 * 1) ** 2, (0.03 * 1) ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)) / \
           ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))

    return rho, ssim


if __name__ == "__main__":
    print("=== Chargement d'un échantillon d'images Figshare ===")
    X_raw, y, masks = load_figshare_mat_files(FIGSHARE_DIR)
    n_total = len(X_raw)
    rng_sample = np.random.RandomState(RNG_SEED)
    has_mask = masks.sum(axis=(1, 2)) > 0
    valid_idx = np.where(has_mask)[0]
    sample_idx = rng_sample.choice(valid_idx, size=min(N_SAMPLE_IMAGES, len(valid_idx)), replace=False)
    print(f"Échantillon : {len(sample_idx)} images (sur {n_total} total)")

    X_sample_raw = X_raw[sample_idx]
    X_sample = np.stack([finalize_image(img) for img in X_sample_raw])

    print(f"\n=== Chargement du modèle entraîné ({WEIGHTS_PATH}) ===")
    model = build_brnet()
    model.load_weights(WEIGHTS_PATH)

    print("\n=== Étape 1 : cartes Grad-CAM ORIGINALES (aucune randomisation) ===")
    preds_orig = model.predict(X_sample, verbose=0)
    target_classes = preds_orig.argmax(axis=1)
    cams_original = []
    for i in range(len(X_sample)):
        cam = grad_cam_fixed_class(model, X_sample[i], int(target_classes[i]))
        cams_original.append(cam)
    cams_original = np.stack(cams_original)
    print("Cartes de référence calculées.")

    print("\n=== Étape 2 : randomisation cascade, couche par couche ===")
    results = []
    rng_random = np.random.RandomState(RNG_SEED)

    for step, layer_name in enumerate(LAYER_ORDER_CASCADE):
        print(f"\n--- Randomisation cumulative jusqu'à '{layer_name}' (étape {step+1}/{len(LAYER_ORDER_CASCADE)}) ---")
        randomize_layer_weights(model, layer_name, rng_random)

        cams_randomized = []
        for i in range(len(X_sample)):
            cam = grad_cam_fixed_class(model, X_sample[i], int(target_classes[i]))
            cams_randomized.append(cam)
        cams_randomized = np.stack(cams_randomized)

        rhos, ssims = [], []
        for i in range(len(X_sample)):
            rho, ssim = cam_similarity(cams_original[i], cams_randomized[i])
            if not np.isnan(rho):
                rhos.append(rho)
            if not np.isnan(ssim):
                ssims.append(ssim)

        mean_rho = np.mean(rhos) if rhos else np.nan
        mean_ssim = np.mean(ssims) if ssims else np.nan
        print(f"Similarité moyenne -- Spearman rho: {mean_rho:.4f}, SSIM approx: {mean_ssim:.4f}")

        results.append({
            "step": step + 1, "layer_randomized": layer_name,
            "cumulative_layers_randomized": ",".join(LAYER_ORDER_CASCADE[:step+1]),
            "mean_spearman_rho": mean_rho, "mean_ssim": mean_ssim,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("xai_sanity_check_cascade.csv", index=False)

    print("\n=== RÉSUMÉ COMPLET ===")
    print(results_df.to_string(index=False))

    print("\n=== Interprétation ===")
    first_step_rho = results_df.iloc[0]["mean_spearman_rho"]
    last_step_rho = results_df.iloc[-1]["mean_spearman_rho"]
    print(f"Similarité après randomisation SEULE dernière couche : rho={first_step_rho:.4f}")
    print(f"Similarité après randomisation TOTALE : rho={last_step_rho:.4f}")
    print(">>> Analyse descriptive uniquement : aucun seuil post-hoc PASS/FAIL n'est appliqué.")
    print(">>> Une conclusion formelle exige plusieurs checkpoints et la conservation des résultats par image.")

    print("\nFichier sauvegardé : xai_sanity_check_cascade.csv")
