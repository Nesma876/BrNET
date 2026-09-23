"""
BrNet -- Benchmark computationnel (Point 15 de l'audit Nesma)
================================================================================
Mesure latence, throughput, dans un environnement contrôlé (même GPU, mêmes
conditions, moyenne sur plusieurs runs).

Nécessite : TensorFlow avec GPU. Les poids entraînés ne sont PAS nécessaires
(la vitesse ne dépend pas des valeurs des poids, seulement de l'architecture).
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB7
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

IMG_SIZE = 256
N_CLASSES = 4
N_WARMUP = 10
N_TIMED_RUNS = 100
BATCH_SIZE_SINGLE = 1
BATCH_SIZE_THROUGHPUT = 32


def build_brnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=N_CLASSES):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)
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
    return models.Model(inputs, outputs)


def build_mobilenetv2(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=N_CLASSES):
    base = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = layers.Lambda(mobilenet_preprocess)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def build_efficientnetb7(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=N_CLASSES):
    base = EfficientNetB7(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def measure_latency_and_throughput(model, input_shape, model_name):
    print(f"\n--- {model_name} ---")

    x_single = np.random.rand(BATCH_SIZE_SINGLE, *input_shape).astype(np.float32) * 255
    for _ in range(N_WARMUP):
        _ = model.predict(x_single, verbose=0)

    latencies = []
    for _ in range(N_TIMED_RUNS):
        start = time.perf_counter()
        _ = model.predict(x_single, verbose=0)
        latencies.append(time.perf_counter() - start)
    latencies = np.array(latencies) * 1000

    x_batch = np.random.rand(BATCH_SIZE_THROUGHPUT, *input_shape).astype(np.float32) * 255
    for _ in range(5):
        _ = model.predict(x_batch, verbose=0)

    throughput_times = []
    for _ in range(20):
        start = time.perf_counter()
        _ = model.predict(x_batch, verbose=0)
        throughput_times.append(time.perf_counter() - start)
    throughput_times = np.array(throughput_times)
    images_per_sec = BATCH_SIZE_THROUGHPUT / throughput_times.mean()

    n_params = model.count_params()

    print(f"Paramètres: {n_params:,}")
    print(f"Latence (batch=1): {latencies.mean():.2f} ms (median={np.median(latencies):.2f}, SD={latencies.std():.2f})")
    print(f"Throughput (batch={BATCH_SIZE_THROUGHPUT}): {images_per_sec:.1f} images/sec")

    return {
        "model": model_name, "n_params": n_params,
        "latency_mean_ms": latencies.mean(), "latency_median_ms": np.median(latencies),
        "latency_sd_ms": latencies.std(),
        "throughput_images_per_sec": images_per_sec,
    }


if __name__ == "__main__":
    print("=== Vérification GPU ===")
    print("GPU disponible:", tf.config.list_physical_devices("GPU"))

    results = []

    print("\n=== BrNet ===")
    model_brnet = build_brnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=N_CLASSES)
    results.append(measure_latency_and_throughput(model_brnet, (IMG_SIZE, IMG_SIZE, 1), "BrNet"))
    del model_brnet
    tf.keras.backend.clear_session()

    print("\n=== MobileNetV2 ===")
    model_mobile = build_mobilenetv2(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=N_CLASSES)
    results.append(measure_latency_and_throughput(model_mobile, (IMG_SIZE, IMG_SIZE, 3), "MobileNetV2"))
    del model_mobile
    tf.keras.backend.clear_session()

    print("\n=== EfficientNetB7 ===")
    model_eff = build_efficientnetb7(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=N_CLASSES)
    results.append(measure_latency_and_throughput(model_eff, (IMG_SIZE, IMG_SIZE, 3), "EfficientNetB7"))
    del model_eff
    tf.keras.backend.clear_session()

    print("\n=== NOTE : ViT-B/16 non mesuré ici (implémentation PyTorch séparée) ===")

    results_df = pd.DataFrame(results)
    results_df.to_csv("compute_benchmark_results.csv", index=False)

    print("\n=== RÉSUMÉ COMPLET ===")
    print(results_df.to_string(index=False))
    print("\nFichier sauvegardé : compute_benchmark_results.csv")
