import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def build_dataset(data_dir: Path, img_size: int, batch_size: int, seed: int):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="validation",
        seed=seed,
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
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    loss, acc, auc = model.evaluate(val_ds, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Validation AUC: {auc:.4f}")

    y_true = np.concatenate([y for _, y in val_ds], axis=0).ravel().astype(int)
    y_pred = model.predict(val_ds, verbose=0)
    y_pred_labels = (y_pred.ravel() >= 0.5).astype(int)

    print(confusion_matrix(y_true, y_pred_labels))
    print(classification_report(y_true, y_pred_labels, target_names=val_ds.class_names))


if __name__ == "__main__":
    main()
