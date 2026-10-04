"""Train and evaluate a CNN on the shared train/validation/test manifest."""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from training.common import (
    ARTIFACTS_DIR,
    load_class_names,
    load_dataset_config,
    load_splits,
    save_metadata,
    save_metrics,
)


IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42


def _read_rgb(path):
    path_value = path.numpy().decode("utf-8")
    with Image.open(path_value) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def make_dataset(frame, data_root, class_names, training=False):
    label_to_id = {name: index for index, name in enumerate(class_names)}
    paths = [str(data_root / path) for path in frame["path"]]
    labels = np.asarray([label_to_id[name] for name in frame["label"]], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image = tf.py_function(_read_rgb, [path], Tout=tf.uint8)
        image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(frame), 2000), seed=SEED)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="training_augmentation",
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            augmentation,
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.keras.utils.set_random_seed(SEED)
    frame = load_splits()
    class_names = load_class_names()
    dataset_config = load_dataset_config()
    data_root = Path(dataset_config["dataset_root"])
    train_frame = frame[frame["split"] == "train"]
    val_frame = frame[frame["split"] == "validation"]
    test_frame = frame[frame["split"] == "test"]

    train_ds = make_dataset(train_frame, data_root, class_names, training=True)
    val_ds = make_dataset(val_frame, data_root, class_names)
    test_ds = make_dataset(test_frame, data_root, class_names)
    model = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks,
        shuffle=False,
    )

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    elapsed = 0.0
    sample_count = 0
    for images, _ in test_ds:
        start = time.perf_counter()
        output = model(images, training=False)
        _ = output.numpy()  # Synchronize execution before stopping the timer.
        elapsed += time.perf_counter() - start
        sample_count += int(images.shape[0])
    inference_ms = elapsed * 1000 / max(1, sample_count)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "cnn_model.keras"
    model.save(model_path)
    save_metrics(
        "cnn", y_true, y_pred, class_names, inference_ms, model_path,
        dataset_config["manifest_sha256"],
    )
    save_metadata(
        class_names,
        dataset_config,
        "cnn",
        {
            "artifact": model_path.name,
            "version": f"cnn-{model_path.stat().st_mtime_ns}",
            "algorithm": "Three-block convolutional neural network",
            "image_size": list(IMAGE_SIZE),
            "epochs_requested": 20,
            "dataset_fingerprint": dataset_config["manifest_sha256"],
        },
    )
    print(f"Saved CNN model to {model_path}")


if __name__ == "__main__":
    main()
