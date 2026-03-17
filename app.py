import os
from datetime import datetime

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

model = None
MODEL_ERROR = None
MODEL_PATH = None
CLASS_NAMES = ["Parasitized", "Uninfected"]


if MODEL_ERROR is None:
    current_dir = os.getcwd()
    print(f"Directory: {current_dir}")

    try:
        candidate_paths = []

        preferred = os.path.join(current_dir, "models", "best_model.keras")
        if os.path.exists(preferred):
            candidate_paths.append(preferred)

        for base_dir in [current_dir, os.path.join(current_dir, "models")]:
            if os.path.isdir(base_dir):
                for f in os.listdir(base_dir):
                    if f.endswith(".keras") or f.endswith(".h5"):
                        candidate_paths.append(os.path.join(base_dir, f))

        # Keep order and remove duplicates.
        seen = set()
        ordered_candidates = []
        for p in candidate_paths:
            if p not in seen:
                ordered_candidates.append(p)
                seen.add(p)

        if ordered_candidates:
            last_error = None
            for candidate in ordered_candidates:
                try:
                    print(f"Trying: {os.path.basename(candidate)}")
                    candidate_model = tf.keras.models.load_model(candidate, compile=False)
                    candidate_model.compile(
                        optimizer="adam",
                        loss="binary_crossentropy",
                        metrics=["accuracy"],
                    )
                    MODEL_PATH = candidate
                    model = candidate_model
                    print(f"Loaded! Input: {model.input_shape}, Output: {model.output_shape}")
                    break
                except Exception as load_err:
                    last_error = str(load_err)
                    continue

            if model is None:
                MODEL_ERROR = last_error or "Could not load any discovered model file"
                print(f"X {MODEL_ERROR}")
        else:
            MODEL_ERROR = "No .keras or .h5 model file found"
            print(f"X {MODEL_ERROR}")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"X {MODEL_ERROR}")


def prepare_image(image_file):
    if model is None:
        return None
    try:
        img = Image.open(image_file)
        if img.mode != "RGB":
            img = img.convert("RGB")

        input_shape = model.input_shape

        if len(input_shape) == 2:
            img = img.resize((64, 64))
            img_array = np.array(img, dtype="float32")
            img_gray = np.mean(img_array, axis=2)
            expected_size = input_shape[1]
            img_flat = img_gray.flatten()
            if len(img_flat) < expected_size:
                img_flat = np.pad(img_flat, (0, expected_size - len(img_flat)))
            else:
                img_flat = img_flat[:expected_size]
            return np.expand_dims(img_flat, axis=0)
        else:
            img_size = (input_shape[1], input_shape[2])
            img = img.resize(img_size)
            img_array = np.array(img, dtype="float32")
            return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Image error: {e}")
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if model is None:
            return render_template("index.html", error="Model not loaded.")

        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        image_file = request.files["image"]
        if image_file.filename == "":
            return render_template("index.html", error="No file selected.")

        img_array = prepare_image(image_file)
        if img_array is None:
            return render_template("index.html", error="Image processing failed.")

        prediction = model.predict(img_array, verbose=0)
        prob = float(prediction[0][0])

        if prob > 0.5:
            pred_class = 1
            confidence = prob * 100
        else:
            pred_class = 0
            confidence = (1 - prob) * 100

        return render_template(
            "result.html",
            label=CLASS_NAMES[pred_class],
            confidence=f"{confidence:.2f}%",
        )

    if os.path.exists("templates/index.html"):
        return render_template("index.html")

    return f'''<!DOCTYPE html>
<html><head><title>Malaria Detection</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:Arial;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px}}
.container{{max-width:900px;margin:0 auto;background:white;border-radius:20px;padding:40px;box-shadow:0 20px 60px rgba(0,0,0,0.3)}}
h1{{color:#333;margin-bottom:30px;text-align:center}}
.status{{padding:12px;border-radius:12px;margin:25px 0;text-align:center}}
.success{{background:#d4edda;color:#155724;border:2px solid #c3e6cb}}
.error{{background:#f8d7da;color:#721c24;border:2px solid #f5c6cb}}
.icon{{font-size:3rem;margin-bottom:15px}}
</style></head><body>
<div class="container">
<h1>Malaria Detection System</h1>
<div class="status success"><div class="icon">✅</div><h2>Server Running</h2><p>Port 5000 active</p></div>
<div class="status {'error' if model is None else 'success'}">
<div class="icon">{'❌' if model is None else '✅'}</div>
<h2>Model: {'NOT LOADED' if model is None else 'LOADED'}</h2>
<p>{MODEL_ERROR if MODEL_ERROR else 'Ready!'}</p></div></div></body></html>'''


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"}), 400

        img_array = prepare_image(file)
        if img_array is None:
            return jsonify({"success": False, "error": "Image processing failed"}), 500

        prediction = model.predict(img_array, verbose=0)
        print(f"Prediction: {prediction}")
        print(f"Shape: {prediction.shape}")

        prob = float(prediction[0][0])
        print(f"Probability value: {prob}")

        if prob > 0.5:
            pred_class = 1
            confidence = prob * 100
        else:
            pred_class = 0
            confidence = (1 - prob) * 100

        result = CLASS_NAMES[pred_class]
        parasitized_prob = (1 - prob) * 100
        uninfected_prob = prob * 100

        print(f"Result: {result}, Confidence: {confidence:.2f}%")

        return jsonify({
            "success": True,
            "prediction": result,
            "confidence": round(confidence, 2),
            "details": {
                "parasitized_probability": round(parasitized_prob, 2),
                "uninfected_probability": round(uninfected_prob, 2),
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "running", "model_loaded": model is not None})


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print("\n" + "=" * 70)
    print(f"{'✅' if model else '❌'} Model: {'LOADED' if model else 'NOT LOADED'}")
    if model:
        print(f"   Input: {model.input_shape}, Output: {model.output_shape}")
    print("🌐 Server: http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000, use_reloader=False)
