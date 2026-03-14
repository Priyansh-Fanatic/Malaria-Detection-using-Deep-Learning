from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf


APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "models" / "best_model.keras"
UPLOAD_DIR = APP_DIR / "uploads"
IMG_SIZE = 224

app = Flask(__name__)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="Please select an image.")

        try:
            image = Image.open(file.stream)
        except OSError:
            return render_template("index.html", error="Invalid image file.")

        input_tensor = preprocess_image(image)
        prob = float(model.predict(input_tensor, verbose=0)[0][0])
        label = "Parasitized" if prob >= 0.5 else "Uninfected"
        confidence = prob if label == "Parasitized" else 1.0 - prob

        return render_template(
            "result.html",
            label=label,
            confidence=f"{confidence * 100:.2f}%",
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
