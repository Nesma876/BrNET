"""
BrNet -- Point 6 : dimensions exactes de chaque couche
================================================================================
Corrige le premier essai qui n'affichait que le wrapper "sequential"
(data_augmentation imbriqué) au lieu du détail couche par couche. Construit
le modèle sans le bloc data_augmentation en tête (qui n'est utile qu'à
l'entraînement) pour que model.summary() et l'itération sur les couches
donnent des dimensions lisibles directement exploitables dans le papier.

Pas de GPU nécessaire -- construction du modèle seule, pas d'entraînement.
"""

from tensorflow.keras import layers, models

IMG_SIZE = 256
N_CLASSES_FIGSHARE = 3
N_CLASSES_BENCHMARK = 4


def build_brnet_inspection(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255, name="rescaling")(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="valid", name="conv1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid", name="conv3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid", name="last_conv")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="dense_out")(x)
    model = models.Model(inputs, outputs)
    return model


def print_full_summary(model, label):
    print(f"\n{'='*80}\n{label}\n{'='*80}")
    model.summary()
    print(f"\n--- Dimensions couche par couche (extraction directe, pour le papier) ---")
    for layer in model.layers:
        try:
            out_shape = layer.output.shape
            print(f"  {layer.name:<15} output_shape={out_shape}")
        except Exception as e:
            print(f"  {layer.name:<15} (impossible d'extraire : {e})")
    print(f"\nTotal params: {model.count_params():,}")


if __name__ == "__main__":
    model_3 = build_brnet_inspection((IMG_SIZE, IMG_SIZE, 1), N_CLASSES_FIGSHARE)
    print_full_summary(model_3, "BrNet -- configuration Figshare (3 classes)")

    model_4 = build_brnet_inspection((IMG_SIZE, IMG_SIZE, 1), N_CLASSES_BENCHMARK)
    print_full_summary(model_4, "BrNet -- configuration benchmark agrégé (4 classes)")

    print(f"\n{'='*80}\nVérification manuelle des dimensions (calcul indépendant)\n{'='*80}")
    size = IMG_SIZE
    print(f"Input: {size}x{size}x1")
    for i, (filters, name) in enumerate([(32, "conv1"), (64, "conv2"), (64, "conv3"), (64, "last_conv")], 1):
        size_conv = size - 2
        size_pool = size_conv // 2
        print(f"Bloc {i} ({name}): conv valid 3x3 -> {size_conv}x{size_conv}x{filters}  "
              f"-> maxpool 2x2 -> {size_pool}x{size_pool}x{filters}")
        size = size_pool
    print(f"\nFeature map finale avant Flatten : {size}x{size}x64 = {size*size*64:,} valeurs")
    print(f"(Ceci doit correspondre exactement à ce que Keras rapporte ci-dessus)")
