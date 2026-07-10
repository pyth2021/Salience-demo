from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd

app = Flask(__name__)


# ---------------------------------------------------------
# MODEL LOCATION
# ---------------------------------------------------------

# Find the folder where this app.py file is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained Gradient Boosting model.
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gradient_boosting_model.pkl"
)


# ---------------------------------------------------------
# LOAD MODEL PACKAGE
# ---------------------------------------------------------

try:
    with open(MODEL_PATH, "rb") as file:
        model_package = pickle.load(file)

except FileNotFoundError as error:
    raise RuntimeError(
        f"Model file was not found at: {MODEL_PATH}"
    ) from error

except Exception as error:
    raise RuntimeError(
        f"Unable to load the Gradient Boosting model: {error}"
    ) from error


# Retrieve the saved model components.
model = model_package["model"]
feature_columns = model_package["feature_columns"]
encoders = model_package["encoders"]
label_encoder = model_package["label_encoder"]

model_name = model_package.get(
    "model_name",
    "Gradient Boosting supervised classifier"
)


# ---------------------------------------------------------
# HOME ENDPOINT
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Salience ML Bot Detection API",
        "api_version": "14-feature-model",
        "model": model_name,
        "feature_count": len(feature_columns),
        "endpoints": {
            "home": "/",
            "health": "/health",
            "predict": "/predict"
        }
    }), 200


# ---------------------------------------------------------
# HEALTH ENDPOINT
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "api_version": "14-feature-model",
        "model": model_name,
        "feature_count": len(feature_columns),
        "features": list(feature_columns)
    }), 200


# ---------------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Invalid or missing JSON body."
        }), 400

    try:
        # These 14 features must match the features used during training.
        input_row = {
            "page_path": data.get(
                "page_path",
                "/"
            ),

            "interaction_type": data.get(
                "interaction_type",
                "normal_browsing"
            ),

            "scroll_depth_category": data.get(
                "scroll_depth_category",
                "medium"
            ),

            "request_interval_seconds": float(
                data.get("request_interval_seconds", 10)
            ),

            "user_agent_category": data.get(
                "user_agent_category",
                "normal_browser"
            ),

            "has_favicon_request": int(
                data.get("has_favicon_request", 1)
            ),

            "requested_robots_txt": int(
                data.get("requested_robots_txt", 0)
            ),

            "pages_per_session": int(
                data.get("pages_per_session", 3)
            ),

            "error_rate": float(
                data.get("error_rate", 0.0)
            ),

            "tls_version": data.get(
                "tls_version",
                "TLS1.3"
            ),

            "cipher_suite_count": int(
                data.get("cipher_suite_count", 15)
            ),

            "extension_count": int(
                data.get("extension_count", 12)
            ),

            "alpn": data.get(
                "alpn",
                "h2"
            ),

            "sni_present": int(
                data.get("sni_present", 1)
            )
        }

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": "One or more numeric features contain invalid values.",
            "details": str(error)
        }), 400


    # Convert the input dictionary into a one-row DataFrame.
    input_df = pd.DataFrame([input_row])

    unknown_categories = {}


    # ---------------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # ---------------------------------------------------------

    for column, encoder in encoders.items():
        if column not in input_df.columns:
            continue

        value = str(input_df.loc[0, column])
        known_classes = [str(item) for item in encoder.classes_]

        if value in known_classes:
            encoded_value = encoder.transform([value])[0]

        else:
            # Use the first known training category when the API receives
            # a category that was not present during model training.
            fallback_value = encoder.classes_[0]
            encoded_value = encoder.transform([fallback_value])[0]

            unknown_categories[column] = {
                "received": value,
                "fallback_used": str(fallback_value)
            }

        input_df.loc[0, column] = encoded_value


    # Ensure columns are in exactly the same order as model training.
    try:
        input_df = input_df[feature_columns]

    except KeyError as error:
        return jsonify({
            "error": "The API features do not match the trained model.",
            "details": str(error)
        }), 500


    # ---------------------------------------------------------
    # RUN GRADIENT BOOSTING PREDICTION
    # ---------------------------------------------------------

    try:
        prediction_encoded = model.predict(input_df)[0]

        ml_prediction = label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        probabilities = model.predict_proba(input_df)[0]

        confidence = round(
            float(max(probabilities)),
            4
        )

        class_probabilities = {}

        for class_index, probability in zip(
            model.classes_,
            probabilities
        ):
            class_name = label_encoder.inverse_transform(
                [class_index]
            )[0]

            class_probabilities[str(class_name)] = round(
                float(probability),
                4
            )

    except Exception as error:
        return jsonify({
            "error": "The model could not complete the prediction.",
            "details": str(error)
        }), 500


    # ---------------------------------------------------------
    # RETURN PREDICTION
    # ---------------------------------------------------------

    response = {
        "ml_prediction": str(ml_prediction),
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "features_used": input_row,
        "model": model_name,
        "feature_count": len(feature_columns),
        "api_version": "14-feature-model",
        "privacy_note": (
            "Only minimized telemetry is used. No passwords, cookies, "
            "tokens, names, emails, private content, exact location, "
            "or raw IP addresses are collected."
        )
    }

    if unknown_categories:
        response["unknown_category_fallbacks"] = unknown_categories

    return jsonify(response), 200


# ---------------------------------------------------------
# LOCAL DEVELOPMENT
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )