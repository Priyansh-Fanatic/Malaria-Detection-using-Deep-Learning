import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns


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


def collect_image_stats(data_dir: Path) -> pd.DataFrame:
    data_root = resolve_dataset_root(data_dir)
    records = []
    for label_name in TARGET_CLASSES:
        label_dir = data_root / label_name
        if not label_dir.is_dir():
            continue
        label = label_name
        for img_path in label_dir.glob("*.png"):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                records.append({
                    "label": label,
                    "path": str(img_path),
                    "width": width,
                    "height": height,
                })
            except OSError:
                continue
    return pd.DataFrame.from_records(records)


def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="label")
    plt.title("Class Distribution")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "class_distribution.png")
    plt.close()


def plot_size_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="width", y="height", hue="label", alpha=0.5)
    plt.title("Image Size Distribution")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "size_distribution.png")
    plt.close()


def plot_samples(df: pd.DataFrame, out_dir: Path, samples_per_class: int = 6) -> None:
    plt.figure(figsize=(10, 6))
    labels = sorted(df["label"].unique())
    for i, label in enumerate(labels):
        subset = df[df["label"] == label].sample(samples_per_class, random_state=42)
        for j, (_, row) in enumerate(subset.iterrows()):
            with Image.open(row["path"]) as img:
                ax = plt.subplot(len(labels), samples_per_class, i * samples_per_class + j + 1)
                ax.imshow(img)
                ax.set_axis_off()
                if j == 0:
                    ax.set_title(label)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "sample_grid.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/cell_images")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)

    df = collect_image_stats(data_dir)
    if df.empty:
        raise SystemExit("No images found. Check the data_dir path.")

    print(df.groupby("label")["path"].count())
    print(df[["width", "height"]].describe())

    plot_class_distribution(df, reports_dir)
    plot_size_distribution(df, reports_dir)
    plot_samples(df, reports_dir)


if __name__ == "__main__":
    main()
