from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import csv
import os
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load trained ML model
# -----------------------------
MODEL_PATH = "models/bot_detection_model.pkl"

with open(MODEL_PATH, "rb") as file:
    model_package = pickle.load(file)

model = model_package["model"]
feature_columns = model_package["feature_columns"]
encoders = model_package["encoders"]
label_encoder = model_package["label_encoder"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # -----------------------------
    # 1. Receive safe telemetry only
    # -----------------------------
    request_interval = float(data.get("request_interval_seconds", 10))
    user_agent_category = data.get("user_agent_category", "normal_browser")
    has_favicon_request = int(data.get("has_favicon_request", 1))
    requested_robots_txt = int(data.get("requested_robots_txt", 0))
    pages_per_session = int(data.get("pages_per_session", 3))
    error_rate = float(data.get("error_rate", 0.0))

    tls_version = data.get("tls_version", "TLS1.3")
    cipher_suite_count = int(data.get("cipher_suite_count", 15))
    extension_count = int(data.get("extension_count", 12))
    alpn = data.get("alpn", "h2")
    sni_present = int(data.get("sni_present", 1))

    # -----------------------------
    # 2. Prepare data for ML model
    # -----------------------------
    input_row = {
        "request_interval_seconds": request_interval,
        "user_agent_category": user_agent_category,
        "has_favicon_request": has_favicon_request,
        "requested_robots_txt": requested_robots_txt,
        "pages_per_session": pages_per_session,
        "error_rate": error_rate,
        "tls_version": tls_version,
        "cipher_suite_count": cipher_suite_count,
        "extension_count": extension_count,
        "alpn": alpn,
        "sni_present": sni_present
    }

    input_df = pd.DataFrame([input_row])

    # Encode text columns using saved encoders
    for column, encoder in encoders.items():
        if column in input_df.columns:
            value = input_df.loc[0, column]

            if value in encoder.classes_:
                input_df[column] = encoder.transform(input_df[column])
            else:
                # If the model sees a new unknown category, use the first known category
                input_df[column] = encoder.transform([encoder.classes_[0]])

    input_df = input_df[feature_columns]

    # -----------------------------
    # 3. ML prediction
    # -----------------------------
    prediction_encoded = model.predict(input_df)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    probabilities = model.predict_proba(input_df)[0]
    confidence = round(float(max(probabilities)), 4)

    # -----------------------------
    # 4. Decide action
    # -----------------------------
    if prediction in ["bad_bot", "scanner"]:
        action = "flag_or_block"
        risk_level = "high"
    elif prediction == "good_bot":
        action = "allow_with_monitoring"
        risk_level = "low"
    else:
        action = "allow"
        risk_level = "low"

    # -----------------------------
    # 5. Save telemetry + prediction to CSV
    # -----------------------------
    os.makedirs("data/raw", exist_ok=True)

    csv_file = "data/raw/telemetry_log.csv"

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "page_path": data.get("page_path", ""),
        "interaction_type": data.get("interaction_type", ""),
        "scroll_depth_category": data.get("scroll_depth_category", ""),
        "request_interval_seconds": request_interval,
        "user_agent_category": user_agent_category,
        "has_favicon_request": has_favicon_request,
        "requested_robots_txt": requested_robots_txt,
        "pages_per_session": pages_per_session,
        "error_rate": error_rate,
        "tls_version": tls_version,
        "cipher_suite_count": cipher_suite_count,
        "extension_count": extension_count,
        "alpn": alpn,
        "sni_present": sni_present,
        "ml_prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "action": action
    }

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    # -----------------------------
    # 6. Return result to browser
    # -----------------------------
    return jsonify({
        "ml_prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "action": action,
        "features_used": input_row,
        "privacy_note": "No passwords, cookies, tokens, or raw IP addresses are collected.",
        "timestamp": datetime.utcnow().isoformat()
    })


if __name__ == "__main__":
    app.run(debug=True)