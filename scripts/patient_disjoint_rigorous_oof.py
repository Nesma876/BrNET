"""
BrNet -- Patient-disjoint CV rigoureux (réponse au reviewer, Points 1 et 2)
=================================================================================
Corrige deux problèmes signalés par le reviewer :

POINT 1 (cohérence numérique) : ce script sauvegarde un fichier UNIQUE de
prédictions out-of-fold (chaque image test évaluée exactement une fois, dans
son propre fold), qui doit servir de source unique pour les analyses dérivées.
La valeur historique de 70.84% a été retirée comme non reconstructible : elle
ne constitue ni une valeur attendue ni un contrôle de validité pour ce script.
Le résultat canonique doit être calculé directement à partir du fichier OOF,
après moyenne des probabilités des trois seeds puis argmax.

POINT 2 (validation interne non patient-disjoint) : le validation_split=0.1
de Keras découpait le train set SANS respecter les patients (fuite possible
entre train et validation interne, même si train/test restaient disjoints).
Corrigé avec GroupShuffleSplit sur les patients du train fold.

Règle d'agrégation des 3 seeds (à documenter explicitement dans le papier) :
pour chaque image test, on MOYENNE les probabilités softmax des 3 seeds
(pas un seed choisi arbitrairement), puis on prend l'argmax de cette moyenne
comme prédiction finale. C'est ce résultat agrégé qui doit être utilisé
partout dans le papier -- Tableau patient-disjoint, XAI, masking.

Sortie principale : oof_predictions_full.csv
  Une ligne par image (3064 lignes), avec :
  - patient_id, fold, true_label
  - prob_seed42_c0/c1/c2, prob_seed43_c0/c1/c2, prob_seed44_c0/c1/c2 (probas brutes)
  - prob_mean_c0/c1/c2 (moyenne des 3 seeds)
  - pred_mean (argmax de la moyenne)
  - has_mask (booléen)
"""

import os
import numpy as np
import pandas as pd
import h5py
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score

FIGSHARE_DIR = "/kaggle/working/figshare_raw"
IMG_SIZE = 256
SEEDS = [42, 43, 44]
N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 32
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]  # encodage réel Figshare (label-1)
N_CLASSES = 3
PRED_FILE = "oof_predictions_full.csv"


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


def build_brnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=N_CLASSES):
    """Architecture réelle BrNet. Pour n_classes=3 (Figshare, pas de classe
    healthy), seule la dernière couche Dense change -- compter les vrais
    params ici avant de les citer dans le papier (répond au Point 6)."""
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


def print_layer_shapes(model):
    """Répond au Point 6 : dimensions EXACTES de chaque couche, à reporter
    telles quelles dans le papier (fini les 14x14 vs 16x16 contradictoires)."""
    print("\n=== Dimensions exactes de chaque couche (pour Section Methods) ===")
    for layer in model.layers:
        try:
            print(f"  {layer.name}: output_shape={layer.output_shape}")
        except AttributeError:
            pass


if __name__ == "__main__":
    print("=== Chargement Figshare ===")
    X_raw, y, pids, masks = load_figshare_mat_files(FIGSHARE_DIR)
    print(f"{len(X_raw)} images, {len(set(pids))} patients, classes: {np.bincount(y)}")
    X = np.stack([finalize_image(img) for img in X_raw])

    tmp_model = build_brnet()
    print_layer_shapes(tmp_model)
    print(f"Total params (n_classes=3): {tmp_model.count_params()}")
    del tmp_model

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(X, y, pids))

    n_total = len(X)
    fold_assignment = np.full(n_total, -1)
    probs_per_seed = {seed: np.zeros((n_total, N_CLASSES)) for seed in SEEDS}

    fold_summary_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_assignment[test_idx] = fold_idx
        train_pids = pids[train_idx]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        inner_train_rel, inner_val_rel = next(gss.split(X[train_idx], y[train_idx], train_pids))
        inner_train_idx = train_idx[inner_train_rel]
        inner_val_idx = train_idx[inner_val_rel]

        assert set(pids[inner_train_idx]).isdisjoint(set(pids[inner_val_idx])), "Fuite validation interne !"
        assert set(pids[train_idx]).isdisjoint(set(pids[test_idx])), "Fuite train/test !"

        print(f"\n{'='*70}\nFOLD {fold_idx+1}/{N_FOLDS}  "
              f"(train={len(inner_train_idx)}, val_interne={len(inner_val_idx)}, test={len(test_idx)}, "
              f"patients_test={len(set(pids[test_idx]))})\n{'='*70}")

        for seed in SEEDS:
            print(f"\n--- Fold {fold_idx+1}, seed {seed} ---")
            tf.keras.utils.set_random_seed(seed)
            model = build_brnet()
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
            print(f"Fold {fold_idx+1} seed {seed}: acc={acc:.4f} macroF1={macro_f1:.4f}")

            model.save_weights(f"brnet_fold{fold_idx+1}_seed{seed}_oof.weights.h5")

            fold_summary_rows.append({
                "fold": fold_idx + 1, "seed": seed,
                "n_train": len(inner_train_idx), "n_val_internal": len(inner_val_idx),
                "n_test": len(test_idx), "n_patients_test": len(set(pids[test_idx])),
                "accuracy": acc, "macro_f1": macro_f1,
            })
            pd.DataFrame(fold_summary_rows).to_csv("fold_seed_summary.csv", index=False)

    print("\n\n=== Agrégation : moyenne des probabilités des 3 seeds ===")
    prob_mean = np.mean([probs_per_seed[s] for s in SEEDS], axis=0)
    pred_mean = prob_mean.argmax(axis=1)

    overall_acc = (pred_mean == y).mean()
    overall_macro_f1 = f1_score(y, pred_mean, average="macro")
    try:
        overall_macro_auc = roc_auc_score(y, prob_mean, multi_class="ovr", average="macro")
    except Exception as e:
        overall_macro_auc = np.nan
        print(f"(macro AUC non calculable: {e})")

    print(f"Accuracy globale (moyenne 3 seeds, pooled out-of-fold) : {overall_acc:.4f}")
    print(f"Macro-F1 globale : {overall_macro_f1:.4f}")
    print(f"Macro-AUC globale (one-vs-rest) : {overall_macro_auc:.4f}")
    print("Résultat calculé directement depuis le registre OOF canonique; aucune valeur historique n'est utilisée comme cible.")

    print("\n=== Accuracy par fold (sur la moyenne des 3 seeds, out-of-fold) ===")
    for fold_idx in range(N_FOLDS):
        idx_fold = np.where(fold_assignment == fold_idx)[0]
        acc_fold = (pred_mean[idx_fold] == y[idx_fold]).mean()
        f1_fold = f1_score(y[idx_fold], pred_mean[idx_fold], average="macro")
        print(f"Fold {fold_idx+1}: n={len(idx_fold)} acc={acc_fold:.4f} macroF1={f1_fold:.4f}")

    has_mask = masks.sum(axis=(1, 2)) > 0
    out_df = pd.DataFrame({
        "image_idx": np.arange(n_total),
        "patient_id": pids,
        "fold": fold_assignment,
        "true_label": y,
        "true_class": [CLASS_NAMES[c] for c in y],
        "has_mask": has_mask,
    })
    for seed in SEEDS:
        for c in range(N_CLASSES):
            out_df[f"prob_seed{seed}_c{c}"] = probs_per_seed[seed][:, c]
    for c in range(N_CLASSES):
        out_df[f"prob_mean_c{c}"] = prob_mean[:, c]
    out_df["pred_mean"] = pred_mean
    out_df["pred_mean_class"] = [CLASS_NAMES[c] for c in pred_mean]
    out_df["correct"] = pred_mean == y

    out_df.to_csv(PRED_FILE, index=False)
    print(f"\nFichier unique de prédictions sauvegardé : {PRED_FILE}")
    print("Ce fichier doit être LA SEULE source pour : accuracy patient-disjoint,")
    print("tableau XAI quantitatif, masking experiment, matched-area control.")

    pd.DataFrame(fold_summary_rows).to_csv("fold_seed_summary.csv", index=False)
    print("Résumé par fold/seed sauvegardé : fold_seed_summary.csv")
