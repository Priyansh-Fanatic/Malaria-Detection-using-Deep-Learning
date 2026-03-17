import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


TARGET_CLASSES = ["Parasitized", "Uninfected"]


def resolve_dataset_root(data_dir: Path) -> Path:
    expected = {"parasitized", "uninfected"}

    def child_dirs(path: Path):
        return {p.name.lower() for p in path.iterdir() if p.is_dir()}

    try:
        root_children = child_dirs(data_dir)
        if root_children == expected:
            return data_dir
    except FileNotFoundError:
        pass

    best = None
    for candidate in data_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        candidate_children = child_dirs(candidate)
        if candidate_children == expected:
            best = candidate
            break

    if best is not None:
        return best

    raise ValueError(
        f"Could not find dataset root with exactly Parasitized and Uninfected under: {data_dir}"
    )


def build_dataset(data_dir: Path, img_size: int, batch_size: int, seed: int):
    data_root = resolve_dataset_root(data_dir)
    print(f"Using dataset root: {data_root}")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root,
        labels="inferred",
        class_names=TARGET_CLASSES,
        label_mode="binary",
        validation_split=0.2,
        subset="validation",
        seed=seed,
        shuffle=True,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    return val_ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/cell_images")
    parser.add_argument("--model_path", type=str, default="models/best_model.keras")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    val_ds = build_dataset(Path(args.data_dir), args.img_size, args.batch_size, args.seed)
    class_names = val_ds.class_names
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    loss, acc, auc = model.evaluate(val_ds, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation AUC: {auc:.4f}")

    y_true_batches = []
    y_pred_batches = []
    for x_batch, y_batch in val_ds:
        y_true_batches.append(y_batch.numpy().ravel().astype(int))
        y_pred_batches.append(model.predict_on_batch(x_batch).ravel())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    y_pred_labels = (y_pred >= 0.5).astype(int)

    print(confusion_matrix(y_true, y_pred_labels))
    print(classification_report(y_true, y_pred_labels, target_names=class_names))


if __name__ == "__main__":
    main()
