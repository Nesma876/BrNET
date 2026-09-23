"""
BrNet — Expérience 3 : étude d'ablation compacte
=================================================
VERSION 3 — architecture corrigée suite à la découverte de xaibt.ipynb :
  - "Full BrNet" = la VRAIE architecture (32-64-64-64 filtres, PAS de
    padding='same', Flatten (pas GAP), PAS de dropout). 895 812 paramètres,
    pas 69 988 comme annoncé initialement dans le manuscrit.
  - La logique des variantes est donc INVERSÉE par rapport à la version
    précédente : puisque BrNet n'a ni GAP ni dropout par défaut, les
    variantes testent l'AJOUT de ces éléments (pas leur retrait), pour voir
    s'ils auraient amélioré le compromis compacité/performance.
  - Filtre médian 3x3 avant Rescaling (appliqué UNE SEULE FOIS, dans le
    modèle — pas de double rescaling comme dans les versions précédentes).
  - 3 seeds d'initialisation par variante -> mean ± SD.

4 variantes :
  1. Full BrNet (référence réelle : Flatten, sans dropout, sans padding='same')
  2. Avec GlobalAveragePooling2D à la place de Flatten (réduit drastiquement
     les paramètres : teste si la compacité peut s'améliorer sans perte)
  3. Avec Dropout(0.3) ajouté
  4. Trois blocs convolutionnels au lieu de quatre

Sortie : results_ablation.csv (une ligne par variante x seed)
         results_ablation_summary.csv (mean ± SD par variante)
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = "/kaggle/input/datasets/rm1000/brain-tumor-mri-scans"
IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 100
N_CLASSES = 4
SPLIT_SEED = 12
MODEL_SEEDS = [42, 43, 44]


def median_filter_batch(x, y):
    def _apply(img):
        img_uint8 = img.numpy().astype(np.uint8)
        filtered = np.stack([cv2.medianBlur(img_uint8[..., c], 3) for c in range(img_uint8.shape[-1])], axis=-1)
        return filtered.astype(np.float32)

    x_filtered = tf.map_fn(lambda img: tf.py_function(_apply, [img], tf.float32), x)
    x_filtered.set_shape(x.shape)
    return x_filtered, y


def build_file_list():
    """Liste tous les fichiers de façon déterministe (triée), indépendante
    de toute session TF -> garantit un split identique même après un
    redémarrage de kernel (filet de sécurité, cf. bug découvert dans
    mcnemar_test.py où le shuffle interne de image_dataset_from_directory
    n'était pas reproductible entre sessions malgré un seed fixe)."""
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
    return idx[:train_size], idx[train_size:train_size + val_size], idx[train_size + val_size:]


def make_dataset_from_paths(filepaths, labels, training):
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img.set_shape((IMG_SIZE, IMG_SIZE, 1))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SPLIT_SEED, reshuffle_each_iteration=False)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds


def load_data():
    filepaths, labels, class_names = build_file_list()
    train_idx, val_idx, test_idx = split_indices(len(filepaths))

    train_ds = make_dataset_from_paths(filepaths[train_idx], labels[train_idx], training=True)
    val_ds = make_dataset_from_paths(filepaths[val_idx], labels[val_idx], training=False)
    test_ds = make_dataset_from_paths(filepaths[test_idx], labels[test_idx], training=False)

    train_ds = train_ds.map(median_filter_batch, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(median_filter_batch, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(median_filter_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
])


def build_brnet(use_gap=False, use_dropout=False, use_padding_same=False, n_blocks=4,
                 input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=N_CLASSES):
    filters = [32, 64, 64, 64][:n_blocks]
    padding = "same" if use_padding_same else "valid"

    inputs = layers.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for f in filters:
        x = layers.Conv2D(f, (3, 3), activation="relu", padding=padding)(x)
        x = layers.MaxPooling2D((2, 2))(x)

    if use_gap:
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.Flatten()(x)

    x = layers.Dense(64, activation="relu")(x)
    if use_dropout:
        x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_eval(name, seed, train_ds, val_ds, test_ds, **kwargs):
    print(f"\n=== {name} (seed={seed}) ===")
    tf.keras.utils.set_random_seed(seed)

    model = build_brnet(**kwargs)
    n_params = model.count_params()

    model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )

    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_pred.extend(probs.argmax(axis=1))
        y_true.extend(y_batch.numpy())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name} (seed={seed}): params={n_params} acc={acc:.4f} macroF1={macro_f1:.4f}")
    return {"Variant": name, "Seed": seed, "Parameters": n_params,
            "Accuracy": round(100 * acc, 2), "Macro-F1": round(100 * macro_f1, 2)}


if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = load_data()
    print("Classes:", class_names)

    variants = [
        dict(name="Full BrNet (Flatten, no dropout, valid padding)",
             use_gap=False, use_dropout=False, use_padding_same=False, n_blocks=4),
        dict(name="With GAP instead of Flatten",
             use_gap=True, use_dropout=False, use_padding_same=False, n_blocks=4),
        dict(name="With Dropout(0.3) added",
             use_gap=False, use_dropout=True, use_padding_same=False, n_blocks=4),
        dict(name="Three convolutional blocks",
             use_gap=False, use_dropout=False, use_padding_same=False, n_blocks=3),
    ]

    RESULTS_PATH = "results_ablation.csv"

    if os.path.exists(RESULTS_PATH):
        done_df = pd.read_csv(RESULTS_PATH)
        done_pairs = set(zip(done_df["Variant"], done_df["Seed"]))
        results = done_df.to_dict("records")
        print(f"Reprise : {len(done_pairs)} runs déjà terminés trouvés dans {RESULTS_PATH}")
    else:
        done_pairs = set()
        results = []

    for v in variants:
        name = v.pop("name")
        for seed in MODEL_SEEDS:
            if (name, seed) in done_pairs:
                print(f"--- {name} (seed={seed}) déjà fait, on saute ---")
                continue
            row = train_and_eval(name, seed, train_ds, val_ds, test_ds, **v)
            results.append(row)
            pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
        v["name"] = name

    df = pd.DataFrame(results)

    summary = df.groupby("Variant").agg(
        Parameters=("Parameters", "first"),
        Accuracy_mean=("Accuracy", "mean"),
        Accuracy_sd=("Accuracy", "std"),
        MacroF1_mean=("Macro-F1", "mean"),
        MacroF1_sd=("Macro-F1", "std"),
    ).reindex([v["name"] for v in variants])

    print("\n=== Table d'ablation finale (mean ± SD sur 3 seeds) ===")
    print(summary.to_string())
    summary.to_csv("results_ablation_summary.csv")
