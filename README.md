# Malaria Detection using Deep Learning

Deep learning pipeline for malaria blood smear image classification with a Flask web app for demo predictions.

The project trains and compares three models:
- `CustomCNN`
- `MobileNetV2`
- `EfficientNetB0`

Target classes:
- `Parasitized`
- `Uninfected`

## Dataset

Download from Kaggle:

https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

Expected folder structure (auto-detected if there is an extra nested `cell_images` folder):

```text
data/raw/cell_images/
  Parasitized/
  Uninfected/
```

Important:
- Keep `data/` local only.
- Do not upload dataset files to GitHub.

## Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Explore Dataset

```powershell
python scripts\explore.py --data_dir data/raw/cell_images
```

Generates dataset visuals/statistics in `reports/`.

## Train Models

Quick run (faster):

```powershell
python scripts\train.py --data_dir data/raw/cell_images --epochs 3
```

Standard run:

```powershell
python scripts\train.py --data_dir data/raw/cell_images --epochs 5 --batch_size 32 --img_size 128 --learning_rate 0.001
```

Training behavior:
- Uses stratified train/validation split (balanced classes).
- Trains all three models sequentially.
- Applies early stopping and restores best weights.
- Saves best checkpoint per model in `models/`.
- Copies overall winner to `models/best_model.keras` for app inference.

## Evaluate

```powershell
python scripts\evaluate.py --data_dir data/raw/cell_images --model_path models/best_model.keras
```

## Reports Generated

During training, per model files are saved to `reports/`:
- `*_history.png` (accuracy/loss curves)
- `*_confusion_matrix.png`
- `*_roc_curve.png` (when both classes are present)

## Run Web App

```powershell
python app.py
```

Open in browser:

http://127.0.0.1:5000

Available endpoints:
- `GET /` - upload UI
- `POST /` - form upload prediction
- `POST /predict` - JSON/file API
- `GET /health` - service health

## Notes

- If you stop training midway, restarting will retrain models from scratch unless you implement checkpoint resume logic.
- TensorFlow warning `OUT_OF_RANGE: End of sequence` may appear after evaluation loops; this is typically informational for dataset iteration end.
- `.gitignore` is configured to exclude virtual environments, datasets, models, reports, and uploads.
