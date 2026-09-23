"""
BrNet — Expérience 4 : McNemar exact + correction de Holm
=============================================================
Entraîne BrNet et les 3 baselines (MobileNetV2, EfficientNetB7, ViT-B/16)
sur EXACTEMENT la même partition du benchmark 7023 images (80/10/10,
seed de split fixe), pour obtenir des prédictions comparables sur le
même test set. Puis compare BrNet à chaque baseline avec le test de
McNemar exact, avec correction de Holm pour les 3 comparaisons.

ATTENTION COÛT DE CALCUL :
  - EfficientNetB7 et ViT-B/16 sont de GROS modèles (65M et 86M paramètres).
    Sur GPU Kaggle (P100/T4), compter significativement plus de temps que
    pour BrNet ou MobileNetV2.
  - Les 4 modèles tournent en SÉQUENCE, avec sauvegarde incrémentale :
    un arrêt ne perd pas ce qui est déjà fait.

Sorties :
  - predictions_common_testset.csv (y_true + prédictions des 4 modèles)
  - results_mcnemar.csv (n01, n10, p exact, p corrigé Holm par comparaison)
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB7
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

DATA_DIR = "/kaggle/input/datasets/rm1000/brain-tumor-mri-scans"
IMG_SIZE_BRNET = 256
IMG_SIZE_TRANSFER = 256
IMG_SIZE_VIT = 224
BATCH_SIZE = 32
EPOCHS = 100
SPLIT_SEED = 12
N_CLASSES = 4
PRED_CSV = "predictions_common_testset.csv"


import cv2


def median_filter_tf(image, label):
    """Applique un flou médian 3x3 IMAGE PAR IMAGE à l'intérieur du batch
    (image_dataset_from_directory retourne des lots, pas des images seules —
    il faut boucler sur la dimension batch, pas traiter le tenseur (N,H,W,C)
    comme une seule image)."""
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
    """Liste tous les fichiers du dataset de façon déterministe (triée),
    indépendante de tout état interne TF/session. C'est la base fixe sur
    laquelle le split train/val/test sera calculé avec numpy, pour garantir
    un split IDENTIQUE d'une session Kaggle à l'autre (contrairement au
    shuffle interne de image_dataset_from_directory, qui n'est pas garanti
    reproductible entre sessions même avec le même seed)."""
    class_names = sorted(os.listdir(DATA_DIR))
    filepaths, labels = [], []
    for label_idx, cls in enumerate(class_names):
        cls_dir = os.path.join(DATA_DIR, cls)
        files = sorted(os.listdir(cls_dir))  # tri alphabétique = ordre déterministe
        for fname in files:
            filepaths.append(os.path.join(cls_dir, fname))
            labels.append(label_idx)
    return np.array(filepaths), np.array(labels), class_names


def split_indices(n_total, seed=SPLIT_SEED):
    """Split déterministe basé sur numpy (PAS sur le shuffle interne de TF),
    reproductible à l'identique quel que soit le nombre de sessions/redémarrages."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n_total)
    train_size = int(0.8 * n_total)
    val_size = int(0.1 * n_total)
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]
    return train_idx, val_idx, test_idx


def make_dataset_from_paths(filepaths, labels, img_size, color_mode, training):
    channels = 1 if color_mode == "grayscale" else 3

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=channels, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        img.set_shape((img_size, img_size, channels))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SPLIT_SEED, reshuffle_each_iteration=False)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds


def load_split(img_size, color_mode="rgb"):
    filepaths, labels, class_names = build_file_list()
    train_idx, val_idx, test_idx = split_indices(len(filepaths))

    train_ds = make_dataset_from_paths(filepaths[train_idx], labels[train_idx], img_size, color_mode, training=True)
    val_ds = make_dataset_from_paths(filepaths[val_idx], labels[val_idx], img_size, color_mode, training=False)
    test_ds = make_dataset_from_paths(filepaths[test_idx], labels[test_idx], img_size, color_mode, training=False)

    train_ds = train_ds.map(median_filter_tf, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(median_filter_tf, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(median_filter_tf, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds


# Augmentation partagée par les 4 modèles. Le papier original prévoyait
# flip horizontal+vertical, mais son propre commentaire de révision note
# que le flip vertical n'est pas anatomiquement plausible pour une IRM
# cérébrale (pas de "cerveau à l'envers") -> on ne garde que l'horizontal,
# cohérent avec cette correction déjà prévue dans le manuscrit.
def make_data_augmentation():
    """Crée une NOUVELLE instance à chaque appel — un objet Sequential partagé
    entre plusieurs build_*() fige sa forme d'entrée au premier usage et casse
    dès qu'un modèle suivant utilise une forme différente (ex. BrNet 256x256x1
    puis ViT-B16 224x224x3)."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
    ])


def get_test_labels(test_ds):
    y_true = []
    for _, labels in test_ds:
        y_true.extend(labels.numpy())
    return np.array(y_true)


def predict_on_test_set(model, test_ds):
    """Calcule labels ET prédictions dans le MÊME passage sur test_ds, pour
    éviter tout risque de désynchronisation lié au reshuffle_each_iteration
    par défaut de tf.data (chaque itération séparée reshuffle différemment)."""
    y_true, preds = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        preds.extend(probs.argmax(axis=1))
        y_true.extend(y_batch.numpy())
    return np.array(y_true), np.array(preds)


def build_brnet(input_shape=(IMG_SIZE_BRNET, IMG_SIZE_BRNET, 1), n_classes=N_CLASSES):
    """Architecture RÉELLE de BrNet (retrouvée dans xaibt.ipynb, create_brnet) :
    32-64-64-64 filtres, PAS de padding='same', Flatten (pas GAP), pas de
    dropout. 895 812 paramètres — pas 69 988 comme annoncé initialement dans
    le manuscrit (chiffre introuvable dans aucun notebook fourni)."""
    inputs = layers.Input(shape=input_shape)
    x = make_data_augmentation()(inputs)
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


def build_mobilenetv2(input_shape=(IMG_SIZE_TRANSFER, IMG_SIZE_TRANSFER, 3), n_classes=N_CLASSES):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = make_data_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_efficientnetb7(input_shape=(IMG_SIZE_TRANSFER, IMG_SIZE_TRANSFER, 3), n_classes=N_CLASSES):
    base = EfficientNetB7(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False  # backbone explicitement figé (même correctif que MobileNetV2)
    inputs = layers.Input(shape=input_shape)
    x = make_data_augmentation()(inputs)
    # PAS de Rescaling ici : EfficientNetB7 (Keras applications) a son propre
    # rescaling interne et attend des pixels bruts 0-255. Appliquer un
    # Rescaling(1/255) en plus double la normalisation et détruit le signal.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_vit_b16(input_shape=(IMG_SIZE_VIT, IMG_SIZE_VIT, 3), n_classes=N_CLASSES):
    """ViT-B/16 pré-entraîné ImageNet via tensorflow_hub, backbone gelé, tête entraînée."""
    import tensorflow_hub as hub
    url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    inputs = layers.Input(shape=input_shape)
    x = make_data_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    vit_layer = hub.KerasLayer(url, trainable=False)
    x = vit_layer(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_predict(model_name, build_fn, img_size, color_mode="rgb"):
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")

    # Libère la mémoire GPU accumulée par les modèles précédents dans cette
    # même session -- évite la saturation mémoire (vue à 15.5/16 GiB avant
    # le blocage sur EfficientNetB7) quand plusieurs gros modèles (65M+
    # paramètres) sont construits successivement sans libération explicite.
    tf.keras.backend.clear_session()

    train_ds, val_ds, test_ds = load_split(img_size, color_mode=color_mode)

    model = build_fn()
    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
    )

    # Sauvegarde des poids (utile pour réutiliser BrNet en validation externe
    # sans devoir tout réentraîner)
    model.save_weights(f"{model_name.lower().replace('-', '_')}_weights_mcnemar.weights.h5")

    y_true, preds = predict_on_test_set(model, test_ds)

    acc = (preds == y_true).mean()
    print(f"{model_name}: accuracy test = {acc:.4f}")
    return y_true, preds


if __name__ == "__main__":
    if os.path.exists(PRED_CSV):
        results_df = pd.read_csv(PRED_CSV)
        print(f"Reprise : colonnes déjà présentes = {list(results_df.columns)}")
    else:
        results_df = None

    models_to_run = [
        ("BrNet", build_brnet, IMG_SIZE_BRNET, "grayscale"),
        ("MobileNetV2", build_mobilenetv2, IMG_SIZE_TRANSFER, "rgb"),
        ("EfficientNetB7", build_efficientnetb7, IMG_SIZE_TRANSFER, "rgb"),
        ("ViT-B16", build_vit_b16, IMG_SIZE_VIT, "rgb"),
    ]

    for name, build_fn, img_size, color_mode in models_to_run:
        col_pred = f"pred_{name}"
        if results_df is not None and col_pred in results_df.columns:
            print(f"--- {name} déjà fait, on saute ---")
            continue

        y_true, preds = train_and_predict(name, build_fn, img_size, color_mode)

        if results_df is None:
            results_df = pd.DataFrame({"y_true": y_true})
        results_df[col_pred] = preds
        results_df.to_csv(PRED_CSV, index=False)
        print(f"Sauvegardé dans {PRED_CSV}")

    print("\n\n=== Toutes les prédictions sont prêtes ===")
    print(results_df.head())

    y_true = results_df["y_true"].values
    pred_brnet = results_df["pred_BrNet"].values
    correct_brnet = (pred_brnet == y_true)

    comparisons = ["MobileNetV2", "EfficientNetB7", "ViT-B16"]
    rows = []
    for baseline in comparisons:
        pred_baseline = results_df[f"pred_{baseline}"].values
        correct_baseline = (pred_baseline == y_true)

        n01 = int(np.sum((~correct_brnet) & correct_baseline))
        n10 = int(np.sum(correct_brnet & (~correct_baseline)))

        result = binomtest(min(n01, n10), n01 + n10, p=0.5, alternative="two-sided") if (n01 + n10) > 0 else None
        p_exact = result.pvalue if result is not None else 1.0

        rows.append({"comparison": f"BrNet vs {baseline}", "n01": n01, "n10": n10, "p_exact": p_exact})
        print(f"BrNet vs {baseline}: n01={n01} n10={n10} p_exact={p_exact:.4f}")

    df_mcnemar = pd.DataFrame(rows)
    _, p_holm, _, _ = multipletests(df_mcnemar["p_exact"].values, method="holm")
    df_mcnemar["p_holm"] = p_holm

    print("\n=== Résultat final McNemar (Holm-corrigé) ===")
    print(df_mcnemar.to_string(index=False))
    df_mcnemar.to_csv("results_mcnemar.csv", index=False)
