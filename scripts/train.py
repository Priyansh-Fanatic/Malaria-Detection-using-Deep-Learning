import argparse
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


TARGET_CLASSES = ["Parasitized", "Uninfected"]

class Config:
    IMG_SIZE = 128
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.001
    SEED = 42

config = Config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEGACY_REPORT_FILES = [
    "warmup_accuracy.png",
    "warmup_loss.png",
    "finetune_accuracy.png",
    "finetune_loss.png",
    "training_log.csv",
]


def configure_accelerator() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No TensorFlow GPU detected. Training will run on CPU.")
        return

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"Using GPU: {gpus}")

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

def build_datasets(data_dir: Path, img_size: int, batch_size: int, seed: int):
    data_root = resolve_dataset_root(data_dir)
    print(f"Using dataset root: {data_root}")

    image_paths = []
    labels = []
    extensions = ("*.png", "*.jpg", "*.jpeg")
    for class_index, class_name in enumerate(TARGET_CLASSES):
        class_dir = data_root / class_name
        for pattern in extensions:
            for img_path in class_dir.glob(pattern):
                image_paths.append(str(img_path))
                labels.append(class_index)

    if not image_paths:
        raise ValueError(f"No images found under {data_root}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    def _build_dataset(paths, y, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
        if training:
            ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

        def _load_image(path, label):
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32)
            label = tf.cast(label, tf.float32)
            return img, label

        ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        return ds

    train_ds = _build_dataset(train_paths, train_labels, training=True)
    val_ds = _build_dataset(val_paths, val_labels, training=False)

    # Quick sanity check to ensure both classes are present in each split.
    train_counts = {0: 0, 1: 0}
    val_counts = {0: 0, 1: 0}
    for _, y_batch in train_ds:
        labels = y_batch.numpy().astype(int).ravel()
        train_counts[0] += int((labels == 0).sum())
        train_counts[1] += int((labels == 1).sum())
    for _, y_batch in val_ds:
        labels = y_batch.numpy().astype(int).ravel()
        val_counts[0] += int((labels == 0).sum())
        val_counts[1] += int((labels == 1).sum())

    print(f"Train split counts: Parasitized={train_counts[0]}, Uninfected={train_counts[1]}")
    print(f"Val split counts: Parasitized={val_counts[0]}, Uninfected={val_counts[1]}")

    return train_ds, val_ds


def resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    # Fall back to project-relative path for clearer error messages later.
    return project_candidate


def remove_stale_report_files(reports_dir: Path) -> None:
    for file_name in LEGACY_REPORT_FILES:
        file_path = reports_dir / file_name
        if file_path.exists():
            file_path.unlink()
            print(f"Removed stale report file: {file_path.name}")

def create_custom_cnn(img_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs, name="CustomCNN")

def create_transfer_model(model_type, img_size):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    
    if model_type == 'MobileNetV2':
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        base_model = tf.keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    elif model_type == 'EfficientNetB0':
        # EfficientNet expects unscaled inputs (0-255)
        x = inputs
        base_model = tf.keras.applications.EfficientNetB0(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs, name=model_type)

def plot_model_history(history, model_name, reports_dir):
    history_dict = history.history or {}
    epochs = range(1, len(history_dict.get("loss", [])) + 1)
    if not epochs:
        print(f"Warning: No history found for {model_name}; skipping history plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"{model_name} Training History")

    train_acc = history_dict.get("accuracy")
    val_acc = history_dict.get("val_accuracy")
    if train_acc is not None:
        ax1.plot(epochs, train_acc, marker="o", linewidth=2, label="Train Accuracy")
    if val_acc is not None:
        ax1.plot(epochs, val_acc, marker="s", linewidth=2, label="Val Accuracy")
    ax1.set_title(f"{model_name} - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(list(epochs))
    ax1.legend()
    ax1.grid(True, alpha=0.35)

    train_loss = history_dict.get("loss")
    val_loss = history_dict.get("val_loss")
    if train_loss is not None:
        ax2.plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
    if val_loss is not None:
        ax2.plot(epochs, val_loss, marker="s", linewidth=2, label="Val Loss")
    ax2.set_title(f"{model_name} - Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(list(epochs))
    ax2.legend()
    ax2.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name}_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_plots(y_true, y_pred_probs, model_name, reports_dir):
    y_pred_labels = (y_pred_probs > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred_labels, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_CLASSES)
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name}_confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig_cm)

    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}", linewidth=2)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"{model_name} - ROC Curve")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports_dir / f"{model_name}_roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig_roc)

def evaluate_model(model, val_dataset):
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    y_true_list = []
    y_pred_probs_list = []

    for x_batch, y_batch in val_dataset:
        y_true_list.append(y_batch.numpy().flatten())
        probs = model(x_batch, training=False).numpy().flatten()
        y_pred_probs_list.append(probs)

    y_true = np.concatenate(y_true_list)
    y_pred_probs = np.concatenate(y_pred_probs_list)
        
    y_pred_labels = (y_pred_probs > 0.5).astype(int)
    
    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred_labels,
        target_names=TARGET_CLASSES,
        zero_division=0,
    )
    print(report)

    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
        print("ROC AUC skipped: validation split has a single class.")
    else:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    metrics = {
        'val_accuracy': np.mean(y_true == y_pred_labels),
        'roc_auc': roc_auc
    }

    return metrics, y_true, y_pred_probs

def train_model(model, model_name, train_gen, val_gen, epochs, learning_rate, model_dir, reports_dir):
    print(f"\n{'='*50}\nTraining {model_name}\n{'='*50}")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    
    filepath = str(model_dir / f"{model_name}_best.keras")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    plot_model_history(history, model_name, reports_dir)
    return history, model

def main():
    configure_accelerator()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/cell_images")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    data_dir = resolve_input_path(args.data_dir)
    model_dir = resolve_input_path(args.model_dir)
    reports_dir = resolve_input_path(args.reports_dir)

    print(f"Resolved data_dir: {data_dir}")
    print(f"Resolved model_dir: {model_dir}")
    print(f"Resolved reports_dir: {reports_dir}")
    
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_report_files(reports_dir)

    train_ds, val_ds = build_datasets(
        data_dir, 
        args.img_size, 
        args.batch_size, 
        config.SEED
    )
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # 3. Train multiple models
    models_to_train = {
        'CustomCNN': create_custom_cnn(args.img_size),
        'MobileNetV2': create_transfer_model('MobileNetV2', args.img_size),
        'EfficientNetB0': create_transfer_model('EfficientNetB0', args.img_size),
    }

    results = {}

    for model_name, model in models_to_train.items():
        print(f"\n{model_name} Summary:")
        model.summary()

        # Train model
        history, trained_model = train_model(
            model, model_name, train_ds, val_ds,
            args.epochs, args.learning_rate,
            model_dir, reports_dir
        )
        
        # Evaluate model
        metrics, y_true, y_pred_probs = evaluate_model(trained_model, val_ds)
        save_evaluation_plots(y_true, y_pred_probs, model_name, reports_dir)
        results[model_name] = metrics

    # 4. Find best model
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)

    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Validation Accuracy': [results[m]['val_accuracy'] for m in results.keys()],
        'ROC AUC': [results[m]['roc_auc'] for m in results.keys()]
    })
    print("\n", comparison_df)

    if not comparison_df.empty:
        best_model_name = comparison_df.loc[comparison_df['Validation Accuracy'].idxmax(), 'Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"Validation Accuracy: {results[best_model_name]['val_accuracy']:.4f}")
        print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
        
        # Optional: save to a generic "best_model.keras" for the Flask app to use seamlessly
        best_model_path = model_dir / f"{best_model_name}_best.keras"
        final_best_path = model_dir / "best_model.keras"
        import shutil
        if best_model_path.exists():
            shutil.copy2(best_model_path, final_best_path)
            print(f"-> Saved general best model for app.py to {final_best_path}")

if __name__ == "__main__":
    main()
