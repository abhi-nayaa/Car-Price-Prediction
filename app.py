
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# --------------------------------------------------
# 1. Find the project and model files
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

# Your app.py is in the main project folder.
PROJECT_DIR = APP_DIR

MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "car_price_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

model = None
metadata = None
load_error = None

# --------------------------------------------------
# 2. Load the trained model and metadata
# --------------------------------------------------

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    print("Trained model loaded successfully!")

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        print("Model metadata loaded successfully!")
    else:
        print("WARNING: model_metadata.json is missing.")

except Exception as error:
    load_error = str(error)
    print("Model loading failed:", load_error)


# --------------------------------------------------
# 3. Check model availability
# --------------------------------------------------

def model_is_ready():
    """Check whether the model and metadata are available."""
    return model is not None and metadata is not None


def get_selected_model():
    """Return the saved model name or the model's class name."""
    if metadata and metadata.get("selected_model"):
        return metadata["selected_model"]

    if model is not None:
        return type(model).__name__

    return "Not available"


# --------------------------------------------------
# 4. Validate prediction input
# --------------------------------------------------

def validate_input(payload):
    """Validate car details before prediction."""

    if not isinstance(payload, dict):
        raise ValueError("Send car details as a JSON object.")

    features = metadata["features"]
    numeric_features = metadata["numerical_features"]
    categorical_features = metadata["categorical_features"]

    missing = [
        name for name in features
        if name not in payload
    ]

    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
        )

    cleaned = {}

    for name in features:
        value = payload[name]

        if value is None or str(value).strip() == "":
            raise ValueError(f"Please enter a value for {name}.")

        if name in numeric_features:
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                raise ValueError(f"{name} must be a number.")

            if not np.isfinite(numeric_value):
                raise ValueError(f"{name} must be a finite number.")

            if name in metadata.get("positive_features", []) and numeric_value < 0:
                raise ValueError(f"{name} cannot be negative.")
            cleaned[name] = numeric_value

        elif name in categorical_features:
            text_value = str(value).strip()
            allowed = metadata.get("categories", {}).get(name, [])

            if allowed and text_value not in allowed:
                raise ValueError(
                    f"Invalid value for {name}. "
                    f"Choose from: {', '.join(allowed)}"
                )

            cleaned[name] = text_value

        else:
            cleaned[name] = value

    return cleaned


# --------------------------------------------------
# 5. Home page
# --------------------------------------------------

@app.route("/")
def home():
    """Display the home page."""

    return render_template(
        "index.html",
        model_ready=model_is_ready(),
        selected_model=get_selected_model(),
        load_error=load_error
    )


# --------------------------------------------------
# 6. Prediction page
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """Display the car price prediction form."""

    if not model_is_ready():
        return (
            "The prediction system is not ready. "
            "Check the terminal for missing model files or metadata.",
            503
        )

    return render_template(
        "predict.html",
        metadata=metadata
    )


# --------------------------------------------------
# 7. Prediction API
# --------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict_api():
    """Predict the price of a car using JSON input."""

    if not model_is_ready():
        return jsonify({
            "error": (
                "The prediction system is not ready. "
                "Check that model_metadata.json exists "
                "inside the models folder."
            )
        }), 503

    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({
            "error": "Please send valid JSON."
        }), 400

    try:
        cleaned = validate_input(payload)

        input_df = pd.DataFrame(
            [cleaned],
            columns=metadata["features"]
        )

        # Predict using the saved model.
        predicted_price = float(model.predict(input_df)[0])

        return jsonify({
            "predicted_price": round(predicted_price, 2),
            "currency": metadata.get("currency", "USD"),
            "model": get_selected_model()
        })

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed.")

        return jsonify({
            "error": (
                "Prediction failed. The input format may not "
                "match the format used during model training."
            )
        }), 500


# --------------------------------------------------
# 8. Health check
# --------------------------------------------------

@app.route("/health")
def health():
    """Check whether the application is ready."""

    if model_is_ready():
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "metadata_loaded": True,
            "selected_model": get_selected_model()
        })

    return jsonify({
        "status": "error",
        "model_loaded": model is not None,
        "metadata_loaded": metadata is not None,
        "selected_model": get_selected_model(),
        "error": load_error or "Model metadata is missing."
    }), 503


# --------------------------------------------------
# 9. Start Flask
# --------------------------------------------------

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "0") == "1")
