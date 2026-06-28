from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join("models", "bot_detection_model.pkl")

with open(MODEL_PATH, "rb") as file:
    model_package = pickle.load(file)

model = model_package["model"]
feature_columns = model_package["feature_columns"]
encoders = model_package["encoders"]
label_encoder = model_package["label_encoder"]


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Salience ML Bot Detection API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_row = {
        "request_interval_seconds": float(data.get("request_interval_seconds", 10)),
        "user_agent_category": data.get("user_agent_category", "normal_browser"),
        "has_favicon_request": int(data.get("has_favicon_request", 1)),
        "requested_robots_txt": int(data.get("requested_robots_txt", 0)),
        "pages_per_session": int(data.get("pages_per_session", 3)),
        "error_rate": float(data.get("error_rate", 0.0)),
        "tls_version": data.get("tls_version", "TLS1.3"),
        "cipher_suite_count": int(data.get("cipher_suite_count", 15)),
        "extension_count": int(data.get("extension_count", 12)),
        "alpn": data.get("alpn", "h2"),
        "sni_present": int(data.get("sni_present", 1))
    }

    input_df = pd.DataFrame([input_row])

    for column, encoder in encoders.items():
        if column in input_df.columns:
            value = input_df.loc[0, column]

            if value in encoder.classes_:
                input_df[column] = encoder.transform(input_df[column])
            else:
                input_df[column] = encoder.transform([encoder.classes_[0]])

    input_df = input_df[feature_columns]

    prediction_encoded = model.predict(input_df)[0]
    ml_prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    probabilities = model.predict_proba(input_df)[0]
    confidence = round(float(max(probabilities)), 4)

    return jsonify({
        "ml_prediction": ml_prediction,
        "confidence": confidence,
        "features_used": input_row,
        "privacy_note": "No passwords, cookies, tokens, or raw IP addresses are collected."
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)