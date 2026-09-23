"""
BrNet -- Correction MobileNetV2 uniquement (benchmark agrégé, réponse audit Nesma)
================================================================================
Corrige le bug de preprocessing MobileNetV2 (Rescaling(1/255) au lieu de
mobilenet_v2.preprocess_input) SEULEMENT pour ce modèle, sur le même split
80/10/10 déterministe (seed=12) que le premier run. Les prédictions BrNet,
EfficientNetB7, ViT-B16 déjà présentes dans predictions_common_testset.csv
restent valides (pas de bug de preprocessing chez elles) et ne sont PAS
recalculées ici.

Sortie : predictions_common_testset_v2.csv
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

DATA_DIR = "/kaggle/input/datasets/zineb1993/brenetrev"
IMG_SIZE_TRANSFER = 256
BATCH_SIZE = 32
EPOCHS = 100
SPLIT_SEED = 12
N_CLASSES = 4
PRED_CSV_OLD = "predictions_common_testset.csv"
PRED_CSV_NEW = "predictions_common_testset_v2.csv"

import cv2


def median_filter_tf(image, label):
    def _apply(batch_images):
        batch_np = batch_images.numpy().astype(np.uint8)
        filtered_batch = np.zeros_like(batch_np, dtype=np.float32)
        for i in range(batch_np.shape[0]):
            img = batch_np[i]
            if img.shape[-1] == 1:
                filtered = cv2.medianBlur(img[..., 0], 3)[..., None]
            else:
                filtered = np.stack([cv2.medianBlur(img[..., c], 3) for c in range(img.shape[-1])], axis=-1)
            filtered_batch[i] = filtered.astype(np.float32)
        return filtered_batch

    image_filtered = tf.py_function(_apply, [image], tf.float32)
    image_filtered.set_shape(image.shape)
    return image_filtered, label


def build_file_list():
    class_names = sorted(os.listdir(DATA_DIR))
    filepaths, labels = [], []
    for label_idx, cls in enumerate(class_names):
        cls_dir = os.path.join(DATA_DIR, cls)
        files = sorted(os.listdir(cls_dir))
        for fname in files:
            filepaths.append(os.path.join(cls_dir, fname))
            labels.append(label_idx)
    return np.array(filepaths), np.array(labels), class_names


def split_indices(n_total, seed=SPLIT_SEED):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    train_size = int(0.8 * n_total)
    val_size = int(0.1 * n_total)
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]
    return train_idx, val_idx, test_idx


def make_dataset_from_paths(filepaths, labels, img_size, training):
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img.set_shape((img_size, img_size, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SPLIT_SEED, reshuffle_each_iteration=False)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds


def load_split(img_size):
    filepaths, labels, class_names = build_file_list()
    train_idx, val_idx, test_idx = split_indices(len(filepaths))

    train_ds = make_dataset_from_paths(filepaths[train_idx], labels[train_idx], img_size, training=True)
    val_ds = make_dataset_from_paths(filepaths[val_idx], labels[val_idx], img_size, training=False)
    test_ds = make_dataset_from_paths(filepaths[test_idx], labels[test_idx], img_size, training=False)

    train_ds = train_ds.map(median_filter_tf, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(median_filter_tf, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(median_filter_tf, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds


def make_data_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
    ])


def build_mobilenetv2(input_shape=(IMG_SIZE_TRANSFER, IMG_SIZE_TRANSFER, 3), n_classes=N_CLASSES):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = make_data_augmentation()(inputs)
    x = layers.Lambda(mobilenet_preprocess, name="mobilenet_preprocess")(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def predict_on_test_set(model, test_ds):
    y_true, preds = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        preds.extend(probs.argmax(axis=1))
        y_true.extend(y_batch.numpy())
    return np.array(y_true), np.array(preds)


if __name__ == "__main__":
    print("=== Entraînement MobileNetV2 (preprocessing corrigé) ===")
    train_ds, val_ds, test_ds = load_split(IMG_SIZE_TRANSFER)

    model = build_mobilenetv2()
    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
    )
    model.save_weights("mobilenetv2_v2_weights_mcnemar.weights.h5")

    y_true, preds_mobilenet = predict_on_test_set(model, test_ds)
    acc = (preds_mobilenet == y_true).mean()
    print(f"\nMobileNetV2 (corrigé) : accuracy test = {acc:.4f}")
    print(f"(Ancien résultat, preprocessing incorrect : 93.74% -- à comparer)")

    if os.path.exists(PRED_CSV_OLD):
        old_df = pd.read_csv(PRED_CSV_OLD)
        assert len(old_df) == len(y_true), "Le split ne correspond pas à l'ancien fichier !"
        assert (old_df["y_true"].values == y_true).all(), "y_true ne correspond pas -- split différent !"
        new_df = old_df.copy()
        new_df["pred_MobileNetV2"] = preds_mobilenet
        print("\nFusionné avec l'ancien CSV (BrNet/EfficientNetB7/ViT-B16 conservés).")
    else:
        new_df = pd.DataFrame({"y_true": y_true, "pred_MobileNetV2": preds_mobilenet})
        print("\nAncien CSV introuvable -- seul MobileNetV2 sauvegardé.")

    new_df.to_csv(PRED_CSV_NEW, index=False)
    print(f"Sauvegardé : {PRED_CSV_NEW}")

    if "pred_BrNet" in new_df.columns:
        from scipy.stats import binomtest
        pred_brnet = new_df["pred_BrNet"].values
        correct_brnet = (pred_brnet == y_true)
        correct_mobile = (preds_mobilenet == y_true)
        n01 = int(np.sum((~correct_brnet) & correct_mobile))
        n10 = int(np.sum(correct_brnet & (~correct_mobile)))
        result = binomtest(min(n01, n10), n01 + n10, p=0.5, alternative="two-sided") if (n01 + n10) > 0 else None
        p_exact = result.pvalue if result is not None else 1.0
        print(f"\nBrNet vs MobileNetV2 (corrigé) : n01={n01} n10={n10} p_exact={p_exact:.4f}")
