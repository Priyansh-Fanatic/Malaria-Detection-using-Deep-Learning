# Malaria Detection using Deep Learning

This project trains a MobileNetV2 model to classify malaria blood smear images as Parasitized or Uninfected and serves predictions via a Flask web app.

## Dataset
Download the dataset from Kaggle (do not upload dataset files/folders to GitHub):

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

After download and extraction, the dataset is expected at:

```
data/raw/cell_images/
  Parasitized/
  Uninfected/
```

This matches the Kaggle malaria dataset structure.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Exploration

```bash
python scripts/explore.py --data_dir data/raw/cell_images
```

This produces plots in `reports/`.

## Training

```bash
python scripts/train.py --data_dir data/raw/cell_images --epochs 15 --fine_tune_epochs 5
```

The best model is saved to `models/best_model.keras`.

## Evaluation

```bash
python scripts/evaluate.py --data_dir data/raw/cell_images --model_path models/best_model.keras
```

## Run the Web App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Notes
- Image augmentation is already applied in the dataset; on-the-fly augmentation is disabled by default in the training script.
- Validation split is handled in `train.py` via `image_dataset_from_directory`.
- Keep `data/` local only. Do not push dataset files to the repository.
