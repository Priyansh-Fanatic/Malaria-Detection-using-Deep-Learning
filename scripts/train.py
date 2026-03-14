import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


def build_datasets(data_dir: Path, img_size: int, batch_size: int, seed: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        validation_split=0.2,
        subset="training",
        seed=seed,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
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
    return train_ds, val_ds


def build_model(img_size: int, use_augmentation: bool):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = inputs
    if use_augmentation:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.05),
            ],
            name="augmentation",
        )
        x = augmentation(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )


def plot_history(history, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_accuracy.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_loss.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/cell_images")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    reports_dir = Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = build_datasets(data_dir, args.img_size, args.batch_size, args.seed)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model, base_model = build_model(args.img_size, args.use_augmentation)
    compile_model(model, lr=1e-3)

    checkpoint_path = model_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2),
        tf.keras.callbacks.CSVLogger(str(reports_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - 20)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    compile_model(model, lr=1e-5)
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs + args.fine_tune_epochs,
        initial_epoch=len(history.epoch),
        callbacks=callbacks,
    )

    plot_history(history, reports_dir, prefix="warmup")
    plot_history(fine_history, reports_dir, prefix="finetune")

    model.save(model_dir / "final_model.keras")


if __name__ == "__main__":
    main()
