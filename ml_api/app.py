from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# This file is a compatibility alias for the Gradient Boosting model.
# train_model.py saves the Gradient Boosting package as bot_detection_model.pkl
# so Azure can load it using the existing deployment path.
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
        "model": model_package.get("model_name", "Unknown model"),
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model": model_package.get("model_name", "Unknown model"),
        "feature_count": len(feature_columns),
        "features": feature_columns
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Invalid or missing JSON body."
        }), 400

    # Must match the 14 FEATURES list used in src/train_model.py
    input_row = {
        "page_path": data.get("page_path", "/"),
        "interaction_type": data.get("interaction_type", "normal_browsing"),
        "scroll_depth_category": data.get("scroll_depth_category", "medium"),
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

    # Encode categorical values using the saved training encoders.
    # If a new unseen category appears, use the first known category
    # to avoid breaking the live API.
    for column, encoder in encoders.items():
        if column in input_df.columns:
            value = str(input_df.loc[0, column])

            if value in encoder.classes_:
                input_df[column] = encoder.transform([value])
            else:
                input_df[column] = encoder.transform([encoder.classes_[0]])

    # Ensure the live input columns match the trained model columns exactly.
    input_df = input_df[feature_columns]

    prediction_encoded = model.predict(input_df)[0]
    ml_prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    probabilities = model.predict_proba(input_df)[0]
    confidence = round(float(max(probabilities)), 4)

    return jsonify({
        "ml_prediction": ml_prediction,
        "confidence": confidence,
        "features_used": input_row,
        "model": model_package.get("model_name", "Gradient Boosting supervised classifier"),
        "feature_count": len(feature_columns),
        "privacy_note": "Only minimized telemetry is used. No passwords, cookies, tokens, names, emails, private content, exact location, or raw IP addresses are collected."
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)