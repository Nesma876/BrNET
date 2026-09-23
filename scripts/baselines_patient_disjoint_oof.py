"""
BrNet -- Baselines sur folds patient-disjoint Figshare (réponse Point 3)
================================================================================
Entraîne MobileNetV2 et EfficientNetB7 sur EXACTEMENT les mêmes 5 folds
patient-disjoint (GroupKFold sur patient ID) que BrNet, avec le même
protocole (validation interne patient-disjointe, 3 seeds/fold, pooling par
moyenne des probabilités). Permet un McNemar patient-disjoint valide entre
BrNet et ces baselines, répondant à l'objection du reviewer sur les
observations corrélées du benchmark agrégé.

IMPORTANT : ViT-B/16 est traité SÉPARÉMENT dans un script PyTorch dédié,
pour la même raison qu'en Section 4 (incompatibilité Keras 3 /
tensorflow_hub sur ce GPU).

Sortie : oof_predictions_baselines.csv
  Même structure que oof_predictions_full.csv (BrNet), pour pouvoir
  fusionner et calculer McNemar patient-disjoint directement.
"""

import os
import numpy as np
import pandas as pd
import h5py
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB7
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score

FIGSHARE_DIR = "/kaggle/working/figshare_raw"
IMG_SIZE = 256
SEEDS = [42, 43, 44]
N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 32
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
N_CLASSES = 3
PRED_FILE = "oof_predictions_mobilenetv2_v2.csv"


def load_figshare_mat_files(mat_dir):
    """Charge en RVB (réplication du canal unique) pour les backbones
    ImageNet-pretrained, qui attendent 3 canaux -- répond aussi au Point 6
    sur la conversion grayscale->RGB."""
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


def finalize_image_rgb(img_uint8):
    """Filtre médian puis réplication sur 3 canaux (RVB) -- PAS de /255 ici,
    chaque backbone applique son propre rescaling en interne."""
    filtered = cv2.medianBlur(img_uint8, 3)
    rgb = np.stack([filtered, filtered, filtered], axis=-1)
    return rgb.astype(np.float32)


def make_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
    ])


def build_mobilenetv2(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=N_CLASSES):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = make_data_augmentation()(inputs)
    # CORRECTIF (audit Nesma) : MobileNetV2 attend son propre preprocessing
    # (pixels normalisés en [-1, 1] via mobilenet_v2.preprocess_input), pas
    # un simple Rescaling(1/255) generique. L'ancien code utilisait
    # Rescaling(1/255), ce qui donne des entrees dans [0,1] au lieu de
    # [-1,1] -- distribution tres differente de ce que le backbone
    # ImageNet-pretrained attend, degradant potentiellement les features
    # extraites. Corrige ici via une Lambda qui appelle explicitement
    # preprocess_input sur l'entree brute en [0,255].
    x = layers.Lambda(mobilenet_preprocess, name="mobilenet_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_efficientnetb7(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=N_CLASSES):
    base = EfficientNetB7(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = make_data_augmentation()(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


MODELS = {
    "MobileNetV2": build_mobilenetv2,
    # EfficientNetB7 retiré : déjà vérifié correct (pas de Rescaling
    # supplémentaire, preprocessing géré nativement par le modèle Keras
    # Applications), pas besoin de le refaire. Seul MobileNetV2 avait le
    # bug de preprocessing signalé par l'audit.
}


if __name__ == "__main__":
    print("=== Chargement Figshare (RGB) ===")
    X_raw, y, pids, masks = load_figshare_mat_files(FIGSHARE_DIR)
    print(f"{len(X_raw)} images, {len(set(pids))} patients, classes: {np.bincount(y)}")
    X = np.stack([finalize_image_rgb(img) for img in X_raw])

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(X, y, pids))

    n_total = len(X)
    fold_assignment = np.full(n_total, -1)

    all_results = {}
    fold_summary_rows = []
    if os.path.exists("fold_seed_summary_baselines_v2.csv"):
        fold_summary_rows = pd.read_csv("fold_seed_summary_baselines_v2.csv").to_dict("records")

    for model_name, build_fn in MODELS.items():
        print(f"\n\n{'#'*70}\n# MODELE : {model_name}\n{'#'*70}")

        RAW_PROBS_FILE = f"oof_probs_{model_name}_raw.npz"
        if os.path.exists(RAW_PROBS_FILE):
            print(f"Reprise depuis {RAW_PROBS_FILE}")
            saved = np.load(RAW_PROBS_FILE)
            probs_per_seed = {seed: saved[f"seed{seed}"] for seed in SEEDS}
            fold_assignment = saved["fold_assignment"]
            done_folds = set(saved["done_folds"].tolist())
        else:
            probs_per_seed = {seed: np.zeros((n_total, N_CLASSES)) for seed in SEEDS}
            done_folds = set()

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if fold_idx in done_folds:
                print(f"Fold {fold_idx+1} déjà fait, on saute.")
                continue

            fold_assignment[test_idx] = fold_idx
            train_pids = pids[train_idx]

            gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
            inner_train_rel, inner_val_rel = next(gss.split(X[train_idx], y[train_idx], train_pids))
            inner_train_idx = train_idx[inner_train_rel]
            inner_val_idx = train_idx[inner_val_rel]

            assert set(pids[inner_train_idx]).isdisjoint(set(pids[inner_val_idx]))
            assert set(pids[train_idx]).isdisjoint(set(pids[test_idx]))

            print(f"\n{'='*70}\n{model_name} -- FOLD {fold_idx+1}/{N_FOLDS}\n{'='*70}")

            for seed in SEEDS:
                print(f"\n--- {model_name}, Fold {fold_idx+1}, seed {seed} ---")
                tf.keras.utils.set_random_seed(seed)
                model = build_fn()
                model.fit(
                    X[inner_train_idx], y[inner_train_idx],
                    validation_data=(X[inner_val_idx], y[inner_val_idx]),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
                )
                probs_test = model.predict(X[test_idx], verbose=0)
                probs_per_seed[seed][test_idx] = probs_test

                preds_test = probs_test.argmax(axis=1)
                acc = (preds_test == y[test_idx]).mean()
                macro_f1 = f1_score(y[test_idx], preds_test, average="macro")
                print(f"{model_name} Fold {fold_idx+1} seed {seed}: acc={acc:.4f} macroF1={macro_f1:.4f}")

                fold_summary_rows.append({
                    "model": model_name, "fold": fold_idx + 1, "seed": seed,
                    "n_train": len(inner_train_idx), "n_val_internal": len(inner_val_idx),
                    "n_test": len(test_idx), "accuracy": acc, "macro_f1": macro_f1,
                })
                pd.DataFrame(fold_summary_rows).to_csv("fold_seed_summary_baselines_v2.csv", index=False)

                del model
                tf.keras.backend.clear_session()

            # --- Sauvegarde incrémentale après chaque fold complet ---
            done_folds.add(fold_idx)
            np.savez(
                RAW_PROBS_FILE,
                fold_assignment=fold_assignment,
                done_folds=np.array(list(done_folds)),
                **{f"seed{seed}": probs_per_seed[seed] for seed in SEEDS},
            )
            print(f"[Sauvegarde intermédiaire après Fold {fold_idx+1} -- {RAW_PROBS_FILE}]")

        prob_mean = np.mean([probs_per_seed[s] for s in SEEDS], axis=0)
        pred_mean = prob_mean.argmax(axis=1)
        overall_acc = (pred_mean == y).mean()
        print(f"\n{model_name} -- Accuracy globale (pooled out-of-fold, 3 seeds) : {overall_acc:.4f}")

        all_results[model_name] = {"prob_mean": prob_mean, "pred_mean": pred_mean}

    out_df = pd.DataFrame({
        "image_idx": np.arange(n_total),
        "patient_id": pids,
        "fold": fold_assignment,
        "true_label": y,
        "true_class": [CLASS_NAMES[c] for c in y],
    })
    for model_name, res in all_results.items():
        out_df[f"pred_{model_name}"] = res["pred_mean"]
        for c in range(N_CLASSES):
            out_df[f"prob_{model_name}_c{c}"] = res["prob_mean"][:, c]

    out_df.to_csv(PRED_FILE, index=False)
    print(f"\nFichier de prédictions baselines sauvegardé : {PRED_FILE}")
    print("A fusionner avec oof_predictions_full.csv (BrNet) sur image_idx pour McNemar patient-disjoint.")
