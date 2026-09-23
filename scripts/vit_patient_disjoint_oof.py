"""
BrNet -- ViT-B/16 sur folds patient-disjoint Figshare (réponse Point 3)
================================================================================
Entraîne ViT-B/16 (PyTorch/timm, backbone gelé) sur EXACTEMENT les mêmes 5
folds patient-disjoint (GroupKFold sur patient ID) que BrNet, avec le même
protocole (validation interne patient-disjointe, 3 seeds/fold, pooling par
moyenne des probabilités).

PyTorch plutôt que TensorFlow/tensorflow_hub : évite le bug Keras 3 déjà
rencontré pour la comparaison sur le benchmark agrégé (Section 4).

Sortie : oof_predictions_vit.csv (même structure que oof_predictions_full.csv
et oof_predictions_baselines.csv, pour fusion et calcul McNemar).
"""

import os
import numpy as np
import pandas as pd
import h5py
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score
import timm

FIGSHARE_DIR = "/kaggle/working/figshare_raw"
IMG_SIZE = 224
SEEDS = [42, 43, 44]
N_FOLDS = 5
EPOCHS = 100
BATCH_SIZE = 32
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
N_CLASSES = 3
LEARNING_RATE = 2e-5
PATIENCE = 15
PRED_FILE = "oof_predictions_vit.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


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
        img_filtered = cv2.medianBlur(img_uint8, 3)
        img_resized = cv2.resize(img_filtered, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        images_raw.append(img_resized)
        labels.append(label - 1)
        pids.append(pid)
        masks.append(mask_resized)

    return np.stack(images_raw), np.array(labels), np.array(pids), np.stack(masks)


class FigshareDataset(Dataset):
    def __init__(self, images, labels, training=False):
        self.images = images
        self.labels = labels
        self.training = training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        if self.training and np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
        rgb = np.stack([img, img, img], axis=-1)
        for c in range(3):
            rgb[..., c] = (rgb[..., c] - MEAN[c]) / STD[c]
        rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(rgb), self.labels[idx]


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)


def train_one_run(train_idx, val_idx, test_idx, X_all, y_all, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = FigshareDataset(X_all[train_idx], y_all[train_idx], training=True)
    val_ds = FigshareDataset(X_all[val_idx], y_all[val_idx], training=False)
    test_ds = FigshareDataset(X_all[test_idx], y_all[test_idx], training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ViTClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                n_val += inputs.size(0)
        val_loss /= n_val

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


if __name__ == "__main__":
    print("=== Chargement Figshare ===")
    X_raw, y, pids, masks = load_figshare_mat_files(FIGSHARE_DIR)
    print(f"{len(X_raw)} images, {len(set(pids))} patients, classes: {np.bincount(y)}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(X_raw, y, pids))

    n_total = len(X_raw)
    fold_assignment = np.full(n_total, -1)

    # --- Sauvegarde incrémentale : reprend les probas déjà calculées si le
    # script a été interrompu (fichier perdu en session précédente) ---
    RAW_PROBS_FILE = "oof_probs_vit_raw.npz"
    if os.path.exists(RAW_PROBS_FILE):
        print(f"Reprise depuis {RAW_PROBS_FILE}")
        saved = np.load(RAW_PROBS_FILE)
        probs_per_seed = {seed: saved[f"seed{seed}"] for seed in SEEDS}
        fold_assignment = saved["fold_assignment"]
        done_folds = set(saved["done_folds"].tolist())
    else:
        probs_per_seed = {seed: np.zeros((n_total, N_CLASSES)) for seed in SEEDS}
        done_folds = set()

    fold_summary_rows = []
    if os.path.exists("fold_seed_summary_vit.csv"):
        fold_summary_rows = pd.read_csv("fold_seed_summary_vit.csv").to_dict("records")

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if fold_idx in done_folds:
            print(f"Fold {fold_idx+1} déjà fait, on saute.")
            continue

        fold_assignment[test_idx] = fold_idx
        train_pids = pids[train_idx]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        inner_train_rel, inner_val_rel = next(gss.split(X_raw[train_idx], y[train_idx], train_pids))
        inner_train_idx = train_idx[inner_train_rel]
        inner_val_idx = train_idx[inner_val_rel]

        assert set(pids[inner_train_idx]).isdisjoint(set(pids[inner_val_idx]))
        assert set(pids[train_idx]).isdisjoint(set(pids[test_idx]))

        print(f"\n{'='*70}\nViT-B/16 -- FOLD {fold_idx+1}/{N_FOLDS} "
              f"(train={len(inner_train_idx)}, val={len(inner_val_idx)}, test={len(test_idx)})\n{'='*70}")

        for seed in SEEDS:
            print(f"\n--- ViT-B/16, Fold {fold_idx+1}, seed {seed} ---")
            probs_test = train_one_run(inner_train_idx, inner_val_idx, test_idx, X_raw, y, seed)
            probs_per_seed[seed][test_idx] = probs_test

            preds_test = probs_test.argmax(axis=1)
            acc = (preds_test == y[test_idx]).mean()
            macro_f1 = f1_score(y[test_idx], preds_test, average="macro")
            print(f"ViT-B/16 Fold {fold_idx+1} seed {seed}: acc={acc:.4f} macroF1={macro_f1:.4f}")

            fold_summary_rows.append({
                "model": "ViT-B16", "fold": fold_idx + 1, "seed": seed,
                "n_train": len(inner_train_idx), "n_val_internal": len(inner_val_idx),
                "n_test": len(test_idx), "accuracy": acc, "macro_f1": macro_f1,
            })
            pd.DataFrame(fold_summary_rows).to_csv("fold_seed_summary_vit.csv", index=False)

        # --- Sauvegarde incrémentale APRÈS CHAQUE FOLD COMPLET (3 seeds) ---
        done_folds.add(fold_idx)
        np.savez(
            RAW_PROBS_FILE,
            fold_assignment=fold_assignment,
            done_folds=np.array(list(done_folds)),
            **{f"seed{seed}": probs_per_seed[seed] for seed in SEEDS},
        )
        print(f"[Sauvegarde intermédiaire après Fold {fold_idx+1} -- {RAW_PROBS_FILE}]")

        # Régénère aussi le CSV final avec ce qu'on a jusqu'ici (permet de
        # récupérer un résultat même si le script s'arrête avant la fin)
        prob_mean_partial = np.mean([probs_per_seed[s] for s in SEEDS], axis=0)
        pred_mean_partial = prob_mean_partial.argmax(axis=1)
        out_df_partial = pd.DataFrame({
            "image_idx": np.arange(n_total), "patient_id": pids, "fold": fold_assignment,
            "true_label": y, "true_class": [CLASS_NAMES[c] for c in y],
            "pred_ViT-B16": pred_mean_partial,
        })
        for c in range(N_CLASSES):
            out_df_partial[f"prob_ViT-B16_c{c}"] = prob_mean_partial[:, c]
        out_df_partial.to_csv(PRED_FILE, index=False)
        print(f"[CSV partiel sauvegardé : {PRED_FILE} -- utilisable même si le script s'arrête maintenant]")

    prob_mean = np.mean([probs_per_seed[s] for s in SEEDS], axis=0)
    pred_mean = prob_mean.argmax(axis=1)
    overall_acc = (pred_mean == y).mean()
    print(f"\nViT-B/16 -- Accuracy globale (pooled out-of-fold, 3 seeds) : {overall_acc:.4f}")

    out_df = pd.DataFrame({
        "image_idx": np.arange(n_total), "patient_id": pids, "fold": fold_assignment,
        "true_label": y, "true_class": [CLASS_NAMES[c] for c in y],
        "pred_ViT-B16": pred_mean,
    })
    for c in range(N_CLASSES):
        out_df[f"prob_ViT-B16_c{c}"] = prob_mean[:, c]

    out_df.to_csv(PRED_FILE, index=False)
    print(f"\nFichier sauvegardé : {PRED_FILE}")
