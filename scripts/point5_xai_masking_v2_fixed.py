"""
BrNet -- Point 5 v2 : corrections audit Nesma (classe cible Grad-CAM + region matched-area fixes)
========================================================================================================
Corrige DEUX bugs identifiés par l'audit de code :

BUG 1 (Grad-CAM) : l'ancienne version calculait class_idx = argmax(preds)
INDEPENDAMMENT pour chaque seed, donc si les 3 seeds ne sont pas d'accord
sur la classe predite pour une image, on moyennait des cartes Grad-CAM
expliquant des classes DIFFERENTES -- non-sens. Corrige : la classe cible
est maintenant FIXEE a la classe poolee (pred_mean, deja connue depuis
oof_predictions_full.csv), la MEME pour les 3 seeds.

BUG 2 (matched-area) : l'ancienne version appelait make_matched_area_mask()
UNE FOIS PAR SEED (avec rng_global qui avance a chaque appel), donc chaque
seed masquait une region temoin DIFFERENTE -- pas de vraie moyenne
coherente sur 3 seeds de la MEME perturbation. Corrige : la region
matched-area est maintenant tiree UNE SEULE FOIS par image (avant la boucle
sur les seeds), puis reutilisee identique pour les 3 seeds.

Nécessite : les 15 fichiers brnet_fold{F}_seed{S}_oof.weights.h5 et
oof_predictions_full.csv (deja disponibles, pas de reentrainement).
"""

import os
import numpy as np
import pandas as pd
import h5py
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models

FIGSHARE_DIR = "/kaggle/working/figshare_raw"
OOF_FILE = "oof_predictions_full.csv"
IMG_SIZE = 256
SEEDS = [42, 43, 44]
N_FOLDS = 5
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
N_CLASSES = 3
IOU_THRESHOLDS = [0.3, 0.5, 0.7]
INPAINT_RADIUS = 5
N_BOOTSTRAP = 2000
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


def inpaint_region(img_uint8, mask):
    mask_bin = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    return cv2.inpaint(img_uint8, mask_dilated, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)


def get_brain_mask(img_uint8):
    _, brain_mask = cv2.threshold(img_uint8, 10, 255, cv2.THRESH_BINARY)
    return brain_mask


def make_matched_area_mask(tumor_mask, brain_mask, rng, max_tries=200):
    ys, xs = np.where(tumor_mask > 0)
    if len(ys) == 0:
        return None, 0
    h, w = tumor_mask.shape
    n_tries = 0
    for _ in range(max_tries):
        n_tries += 1
        dy = rng.randint(-h // 2, h // 2)
        dx = rng.randint(-w // 2, w // 2)
        new_ys = ys + dy
        new_xs = xs + dx
        if new_ys.min() < 0 or new_ys.max() >= h or new_xs.min() < 0 or new_xs.max() >= w:
            continue
        candidate = np.zeros_like(tumor_mask)
        candidate[new_ys, new_xs] = 1
        if not np.all(brain_mask[new_ys, new_xs] > 0):
            continue
        if np.any(tumor_mask[new_ys, new_xs] > 0):
            continue
        return candidate, n_tries
    return None, n_tries


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


def grad_cam_fixed_class(model, img, target_class, layer_name="last_conv"):
    """CORRIGE (Bug 1) : la classe cible est maintenant un PARAMETRE fixe
    (target_class), plus calculee en interne par argmax -- garantit que les
    3 seeds expliquent tous la MEME classe (la classe poolee)."""
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


def localization_metrics(cam, mask, thresholds=IOU_THRESHOLDS):
    mask_bin = (mask > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return None
    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    pointing_hit = int(mask_bin[py, px] == 1)
    saliency_mass = cam[mask_bin == 1].sum() / (cam.sum() + 1e-8)
    ious = {}
    for t in thresholds:
        cam_bin = (cam >= t).astype(np.uint8)
        inter = np.logical_and(cam_bin, mask_bin).sum()
        union = np.logical_or(cam_bin, mask_bin).sum()
        ious[t] = inter / union if union > 0 else 0.0
    return pointing_hit, saliency_mass, ious


def patient_clustered_bootstrap_ci(values, patient_ids, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    boot_means = []
    patient_to_indices = {p: np.where(patient_ids == p)[0] for p in unique_patients}

    for _ in range(n_boot):
        sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        idx = np.concatenate([patient_to_indices[p] for p in sampled_patients])
        boot_means.append(np.mean(values[idx]))

    boot_means = np.array(boot_means)
    return np.mean(values), np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)


def paired_bootstrap_diff_ci(values_a, values_b, patient_ids, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """NOUVEAU (répond au point 10 de l'audit) : bootstrap patient-clustered
    de la DIFFERENCE APPARIEE tumor - control, plutot que deux IC separes."""
    diff = values_a - values_b
    return patient_clustered_bootstrap_ci(diff, patient_ids, n_boot, seed)


if __name__ == "__main__":
    print("=== Chargement Figshare + fichier out-of-fold ===")
    X_raw, y, pids, masks = load_figshare_mat_files(FIGSHARE_DIR)
    X = np.stack([finalize_image(img) for img in X_raw])
    oof_df = pd.read_csv(OOF_FILE)
    assert len(oof_df) == len(X), "Le fichier OOF ne correspond pas au nombre d'images chargées !"
    fold_assignment = oof_df["fold"].values
    pred_mean = oof_df["pred_mean"].values
    has_mask = masks.sum(axis=(1, 2)) > 0

    print(f"{len(X)} images, {len(set(pids))} patients")
    print(f"Correctement classifiées (pooled) : {(pred_mean == y).sum()}/{len(y)}")
    print(f"Avec masque disponible : {has_mask.sum()}/{len(y)}")

    rng_global = np.random.RandomState(RNG_SEED)

    print("\n=== ETAPE 1 : Grad-CAM (classe cible = classe poolee FIXE) ===")

    xai_rows = []
    for fold_idx in range(N_FOLDS):
        idx_fold = np.where((fold_assignment == fold_idx) & has_mask)[0]
        print(f"\nFold {fold_idx+1}: {len(idx_fold)} images eligibles (masque dispo)")

        cams_per_seed = {seed: {} for seed in SEEDS}
        for seed in SEEDS:
            weights_path = f"brnet_fold{fold_idx+1}_seed{seed}_oof.weights.h5"
            model = build_brnet()
            model.load_weights(weights_path)
            for i in idx_fold:
                target_class = int(pred_mean[i])
                cam = grad_cam_fixed_class(model, X[i], target_class, "last_conv")
                cams_per_seed[seed][i] = cam
            del model
            tf.keras.backend.clear_session()

        for i in idx_fold:
            cam_avg = np.mean([cams_per_seed[s][i] for s in SEEDS], axis=0)
            metrics = localization_metrics(cam_avg, masks[i])
            if metrics is None:
                continue
            pointing_hit, saliency_mass, ious = metrics

            mask_bin = (masks[i] > 0).astype(np.uint8)
            rand_y, rand_x = rng_global.randint(0, IMG_SIZE), rng_global.randint(0, IMG_SIZE)
            chance_hit = int(mask_bin[rand_y, rand_x] == 1)
            center_hit = int(mask_bin[IMG_SIZE // 2, IMG_SIZE // 2] == 1)
            brain_mask_i = get_brain_mask(X_raw[i])
            brain_ys, brain_xs = np.where(brain_mask_i > 0)
            if len(brain_ys) > 0:
                ridx = rng_global.randint(0, len(brain_ys))
                brain_hit = int(mask_bin[brain_ys[ridx], brain_xs[ridx]] == 1)
            else:
                brain_hit = 0

            row = {
                "image_idx": i, "patient_id": pids[i], "fold": fold_idx,
                "class": CLASS_NAMES[y[i]], "correct": bool(pred_mean[i] == y[i]),
                "pointing_hit": pointing_hit, "saliency_mass": saliency_mass,
                "chance_hit": chance_hit, "center_hit": center_hit, "brain_region_hit": brain_hit,
            }
            for t, v in ious.items():
                row[f"iou@{t}"] = v
            xai_rows.append(row)

    xai_df = pd.DataFrame(xai_rows)
    xai_df.to_csv("results_xai_full_stratified_v2.csv", index=False)

    print("\n=== Resume XAI par classe ET stratifie correct/incorrect (v2, classe cible fixee) ===")
    for stratum in [True, False]:
        sub = xai_df[xai_df["correct"] == stratum]
        label = "CORRECTEMENT classifiees" if stratum else "INCORRECTEMENT classifiees"
        print(f"\n--- {label} (n={len(sub)}) ---")
        if len(sub) > 0:
            summary = sub.groupby("class").agg(
                n=("class", "size"), pointing_hit=("pointing_hit", "mean"),
                saliency_mass=("saliency_mass", "mean"),
                **{f"iou@{t}": (f"iou@{t}", "mean") for t in IOU_THRESHOLDS}
            )
            print(summary)

    print("\n=== Baselines de comparaison (toutes images, macro mean) ===")
    print(f"Pointing-game mesure (Grad-CAM, v2)  : {xai_df['pointing_hit'].mean():.4f}")
    print(f"Baseline chance (pixel aleatoire): {xai_df['chance_hit'].mean():.4f}")
    print(f"Baseline center (toujours centre): {xai_df['center_hit'].mean():.4f}")
    print(f"Baseline brain-region (aleatoire dans cerveau): {xai_df['brain_region_hit'].mean():.4f}")

    mean_pg, lo_pg, hi_pg = patient_clustered_bootstrap_ci(
        xai_df["pointing_hit"].values, xai_df["patient_id"].values
    )
    print(f"\nPointing-game global : {mean_pg:.4f} (IC95% patient-clustered: [{lo_pg:.4f}, {hi_pg:.4f}])")

    print("\n\n=== ETAPE 2 : Masking + matched-area control (region FIXE par image) ===")

    mask_rows = []
    n_matched_failed = 0
    tries_log = []

    for fold_idx in range(N_FOLDS):
        idx_fold = np.where((fold_assignment == fold_idx) & has_mask & (pred_mean == y))[0]
        print(f"\nFold {fold_idx+1}: {len(idx_fold)} images (correctement classifiees, masque dispo)")

        # CORRIGE (Bug 2) : region tiree UNE SEULE FOIS par image, AVANT la
        # boucle sur les seeds.
        matched_masks_per_image = {}
        for i in idx_fold:
            tumor_mask = masks[i]
            brain_mask_i = get_brain_mask(X_raw[i])
            matched_mask, n_tries = make_matched_area_mask(tumor_mask, brain_mask_i, rng_global)
            tries_log.append(n_tries)
            if matched_mask is None:
                n_matched_failed += 1
            matched_masks_per_image[i] = matched_mask

        probs_orig_per_seed = {}
        probs_tumor_per_seed = {}
        probs_matched_per_seed = {}

        for seed in SEEDS:
            weights_path = f"brnet_fold{fold_idx+1}_seed{seed}_oof.weights.h5"
            model = build_brnet()
            model.load_weights(weights_path)

            for i in idx_fold:
                img_raw = X_raw[i]
                tumor_mask = masks[i]

                img_orig = finalize_image(img_raw)
                prob_orig = model.predict(img_orig[None, ...], verbose=0)[0]

                img_tumor_masked = finalize_image(inpaint_region(img_raw, tumor_mask))
                prob_tumor = model.predict(img_tumor_masked[None, ...], verbose=0)[0]

                probs_orig_per_seed.setdefault(i, {})[seed] = prob_orig
                probs_tumor_per_seed.setdefault(i, {})[seed] = prob_tumor

                matched_mask = matched_masks_per_image[i]
                if matched_mask is not None:
                    img_matched = finalize_image(inpaint_region(img_raw, matched_mask))
                    prob_matched = model.predict(img_matched[None, ...], verbose=0)[0]
                    probs_matched_per_seed.setdefault(i, {})[seed] = prob_matched

            del model
            tf.keras.backend.clear_session()

        for i in idx_fold:
            prob_orig_mean = np.mean([probs_orig_per_seed[i][s] for s in SEEDS], axis=0)
            prob_tumor_mean = np.mean([probs_tumor_per_seed[i][s] for s in SEEDS], axis=0)
            pred_orig = int(prob_orig_mean.argmax())
            pred_tumor = int(prob_tumor_mean.argmax())

            has_matched = i in probs_matched_per_seed and len(probs_matched_per_seed[i]) == len(SEEDS)
            if has_matched:
                prob_matched_mean = np.mean([probs_matched_per_seed[i][s] for s in SEEDS], axis=0)
                pred_matched = int(prob_matched_mean.argmax())
                delta_prob_tumor = prob_orig_mean[pred_orig] - prob_tumor_mean[pred_orig]
                delta_prob_matched = prob_orig_mean[pred_orig] - prob_matched_mean[pred_orig]
                changed_tumor = pred_orig != pred_tumor
                changed_matched = pred_orig != pred_matched
            else:
                pred_matched = None
                delta_prob_matched = np.nan
                changed_matched = np.nan
                delta_prob_tumor = prob_orig_mean[pred_orig] - prob_tumor_mean[pred_orig]
                changed_tumor = pred_orig != pred_tumor

            mask_rows.append({
                "image_idx": i, "patient_id": pids[i], "fold": fold_idx,
                "class": CLASS_NAMES[y[i]],
                "pred_orig": pred_orig, "pred_tumor_masked": pred_tumor, "pred_matched_masked": pred_matched,
                "delta_prob_tumor": delta_prob_tumor, "delta_prob_matched": delta_prob_matched,
                "changed_tumor": changed_tumor, "changed_matched": changed_matched,
            })

    mask_df = pd.DataFrame(mask_rows)
    mask_df.to_csv("results_masking_full_patient_clustered_v2.csv", index=False)

    print(f"\n(Matched-area control : {n_matched_failed} images exclues -- aucun placement valide)")
    print(f"Tentatives de placement (moyenne): {np.mean(tries_log):.1f}, max autorise: 200")

    valid_matched = mask_df.dropna(subset=["delta_prob_matched"])

    print("\n=== Resultats agreges (population complete, patient-clustered CI) ===")
    for col, label in [("changed_tumor", "Tumor masked -- %changed"),
                        ("delta_prob_tumor", "Tumor masked -- Delta-prob")]:
        vals = mask_df[col].astype(float).values
        mean_v, lo, hi = patient_clustered_bootstrap_ci(vals, mask_df["patient_id"].values)
        print(f"{label}: {mean_v:.4f} (IC95% patient-clustered: [{lo:.4f}, {hi:.4f}])")

    for col, label in [("changed_matched", "Matched-area -- %changed"),
                        ("delta_prob_matched", "Matched-area -- Delta-prob")]:
        vals = valid_matched[col].astype(float).values
        mean_v, lo, hi = patient_clustered_bootstrap_ci(vals, valid_matched["patient_id"].values)
        print(f"{label}: {mean_v:.4f} (IC95% patient-clustered: [{lo:.4f}, {hi:.4f}])")

    print("\n=== NOUVEAU : comparaison APPARIEE tumor vs control (bootstrap sur la difference) ===")
    changed_tumor_valid = valid_matched["changed_tumor"].astype(float).values
    changed_matched_valid = valid_matched["changed_matched"].astype(float).values
    mean_diff, lo_diff, hi_diff = paired_bootstrap_diff_ci(
        changed_tumor_valid, changed_matched_valid, valid_matched["patient_id"].values
    )
    print(f"Difference appariee (%changed tumor - %changed control) : {mean_diff:.4f} "
          f"(IC95% patient-clustered: [{lo_diff:.4f}, {hi_diff:.4f}])")

    delta_prob_tumor_valid = valid_matched["delta_prob_tumor"].astype(float).values
    delta_prob_matched_valid = valid_matched["delta_prob_matched"].astype(float).values
    mean_diff_p, lo_diff_p, hi_diff_p = paired_bootstrap_diff_ci(
        delta_prob_tumor_valid, delta_prob_matched_valid, valid_matched["patient_id"].values
    )
    print(f"Difference appariee (Delta-prob tumor - Delta-prob control) : {mean_diff_p:.4f} "
          f"(IC95% patient-clustered: [{lo_diff_p:.4f}, {hi_diff_p:.4f}])")

    print("\nFichiers sauvegardes : results_xai_full_stratified_v2.csv, "
          "results_masking_full_patient_clustered_v2.csv")
