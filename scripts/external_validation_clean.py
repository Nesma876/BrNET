"""
BrNet -- Option A : validation externe recalculée sur les images "propres"
================================================================================
Suite à la découverte via perceptual hashing (Point 4) que 685/1137 images
du dataset externe ont un quasi-duplicat dans le benchmark d'entraînement
(distance de Hamming <= 10, confirmé visuellement comme des images
identiques ou quasi-identiques), ce script :

  1. Identifie les images externes SANS quasi-duplicat détecté (dataset
     "propre", véritablement indépendant).
  2. Recharge le modèle BrNet entraîné sur le benchmark agrégé.
  3. Réévalue UNIQUEMENT sur ces images propres.
  4. Compare ce nouveau résultat au chiffre original (80.74% sur les 1137
     images, contaminé) pour quantifier l'effet du chevauchement.

Nécessite : results_phash_duplicates.csv (Point 4, déjà généré),
brnet_weights_mcnemar.weights.h5 (poids BrNet déjà entraînés).
"""

import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

EXTERNAL_DIR = ("/kaggle/input/datasets/souaadrahmoun/brnetval/"
                "Multi-Class Brain Tumor MRI Dataset Glioma, Health/"
                "Brain Cancer MRI Dataset/Brain Cancer MRI Dataset/Brain Cancer MRI Dataset")
DUPLICATES_FILE = "/kaggle/input/datasets/souaadrahmoun/lastupload/results_phash_duplicates.csv"
WEIGHTS_PATH = "/kaggle/input/datasets/souaadrahmoun/mcnemarbrent/brnet_weights_mcnemar.weights.h5"
IMG_SIZE = 256

CLASS_MAPPING = {
    "Glioma": "glioma",
    "Healthy": "healthy",
    "Meningioma": "meningioma",
    "Pituitary Macroadenoma": "pituitary",
}
BRNET_CLASS_ORDER = ["glioma", "healthy", "meningioma", "pituitary"]


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
])


def build_brnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=4):
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


if __name__ == "__main__":
    print("=== Étape 1 : identifier les images externes SANS quasi-duplicat ===")
    dup_df = pd.read_csv(DUPLICATES_FILE)
    contaminated_paths = set(dup_df["external_path"].unique())
    print(f"Images externes contaminées (quasi-duplicat détecté) : {len(contaminated_paths)}")

    all_external_files = []
    for cls_folder in CLASS_MAPPING:
        folder = os.path.join(EXTERNAL_DIR, cls_folder)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            all_external_files.append((fpath, CLASS_MAPPING[cls_folder]))

    print(f"Total images externes trouvées : {len(all_external_files)}")

    clean_files = [(p, c) for p, c in all_external_files if p not in contaminated_paths]
    print(f"Images externes PROPRES (sans quasi-duplicat) : {len(clean_files)}")

    print("\n=== Répartition des images propres par classe ===")
    clean_df_check = pd.DataFrame(clean_files, columns=["path", "class"])
    print(clean_df_check["class"].value_counts())

    print("\n=== Étape 2 : chargement + prédiction du modèle BrNet sur les images propres ===")
    model = build_brnet()
    model.load_weights(WEIGHTS_PATH)
    print(f"Poids chargés depuis {WEIGHTS_PATH}")

    y_true, y_pred, paths_used = [], [], []
    for fpath, true_class in clean_files:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  echec chargement: {fpath}")
            continue
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        # CORRECTIF (audit Nesma) : filtre median manquant -- BrNet a ete
        # entraine avec medianBlur(3) applique systematiquement, l'evaluer
        # sans ce filtre cree un ecart train/test (distribution mismatch).
        img_filtered = cv2.medianBlur(img_resized, 3)
        img_final = img_filtered.astype(np.float32)[..., None]

        probs = model.predict(img_final[None, ...], verbose=0)[0]
        pred_class = BRNET_CLASS_ORDER[probs.argmax()]

        y_true.append(true_class)
        y_pred.append(pred_class)
        paths_used.append(fpath)

    print(f"\n{len(y_true)} images évaluées avec succès")

    acc_clean = accuracy_score(y_true, y_pred)
    f1_clean = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 70)
    print("RÉSULTAT -- VALIDATION EXTERNE SUR IMAGES PROPRES (SANS DUPLICATS)")
    print("=" * 70)
    print(f"Accuracy : {acc_clean:.4f}")
    print(f"Macro-F1 : {f1_clean:.4f}")
    print(f"\nRésultat rétrospectif sur le sous-ensemble filtré :")
    print(f"  {sum(np.asarray(y_true) == np.asarray(y_pred))}/{len(y_true)} images correctement classifiées")
    print(f"  Accuracy avec filtre médian 3x3 : {acc_clean*100:.2f}%")
    print("  Cette sortie ne constitue pas une validation externe indépendante sans provenance patient/site/scanner.")

    print("\n--- Rapport de classification détaillé ---")
    print(classification_report(y_true, y_pred, labels=BRNET_CLASS_ORDER))

    print("\n--- Matrice de confusion ---")
    cm = confusion_matrix(y_true, y_pred, labels=BRNET_CLASS_ORDER)
    cm_df = pd.DataFrame(cm, index=BRNET_CLASS_ORDER, columns=BRNET_CLASS_ORDER)
    print(cm_df)

    pd.DataFrame({
        "path": paths_used, "true_class": y_true, "pred_class": y_pred,
    }).to_csv("results_external_validation_clean.csv", index=False)
    print("\nDétail sauvegardé dans results_external_validation_clean.csv")
