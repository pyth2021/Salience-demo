from flask import Flask, request, jsonify
import os
import pickle

import pandas as pd


app = Flask(__name__)


# ---------------------------------------------------------
# MODEL LOCATIONS
# ---------------------------------------------------------

# Find the folder where this app.py file is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Gradient Boosting supervised model.
GRADIENT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gradient_boosting_model.pkl",
)

# Isolation Forest unsupervised model.
ISOLATION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "isolation_forest_model.pkl",
)


# ---------------------------------------------------------
# LOAD MODEL PACKAGES
# ---------------------------------------------------------

def load_model_package(model_path, model_description):
    """Load a saved model package from a pickle file."""

    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)

    except FileNotFoundError as error:
        raise RuntimeError(
            f"{model_description} file was not found at: {model_path}"
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"Unable to load {model_description}: {error}"
        ) from error


gradient_package = load_model_package(
    GRADIENT_MODEL_PATH,
    "Gradient Boosting model",
)

isolation_package = load_model_package(
    ISOLATION_MODEL_PATH,
    "Isolation Forest model",
)


# ---------------------------------------------------------
# RETRIEVE GRADIENT BOOSTING COMPONENTS
# ---------------------------------------------------------

gradient_model = gradient_package["model"]
feature_columns = gradient_package["feature_columns"]
encoders = gradient_package["encoders"]
label_encoder = gradient_package["label_encoder"]

gradient_model_name = gradient_package.get(
    "model_name",
    "Gradient Boosting supervised classifier",
)


# ---------------------------------------------------------
# RETRIEVE ISOLATION FOREST COMPONENTS
# ---------------------------------------------------------

isolation_model = isolation_package["model"]

isolation_feature_columns = isolation_package.get(
    "feature_columns",
    feature_columns,
)

isolation_model_name = isolation_package.get(
    "model_name",
    "Isolation Forest unsupervised anomaly detector",
)


# Confirm both models use the same feature order.
if list(feature_columns) != list(isolation_feature_columns):
    raise RuntimeError(
        "Gradient Boosting and Isolation Forest feature columns do not match."
    )


# ---------------------------------------------------------
# HOME ENDPOINT
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Salience ML Bot and Anomaly Detection API",
        "api_version": "dual-model-14-feature-api",
        "models": {
            "supervised": gradient_model_name,
            "unsupervised": isolation_model_name,
        },
        "feature_count": len(feature_columns),
        "endpoints": {
            "home": "/",
            "health": "/health",
            "predict": "/predict",
        },
    }), 200


# ---------------------------------------------------------
# HEALTH ENDPOINT
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "models_loaded": {
            "gradient_boosting": True,
            "isolation_forest": True,
        },
        "api_version": "dual-model-14-feature-api",
        "models": {
            "supervised": gradient_model_name,
            "unsupervised": isolation_model_name,
        },
        "feature_count": len(feature_columns),
        "features": list(feature_columns),
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
        # These 14 features must match the training feature list.
        input_row = {
            "page_path": data.get(
                "page_path",
                "/",
            ),

            "interaction_type": data.get(
                "interaction_type",
                "normal_browsing",
            ),

            "scroll_depth_category": data.get(
                "scroll_depth_category",
                "medium",
            ),

            "request_interval_seconds": float(
                data.get(
                    "request_interval_seconds",
                    10,
                )
            ),

            "user_agent_category": data.get(
                "user_agent_category",
                "normal_browser",
            ),

            "has_favicon_request": int(
                data.get(
                    "has_favicon_request",
                    1,
                )
            ),

            "requested_robots_txt": int(
                data.get(
                    "requested_robots_txt",
                    0,
                )
            ),

            "pages_per_session": int(
                data.get(
                    "pages_per_session",
                    3,
                )
            ),

            "error_rate": float(
                data.get(
                    "error_rate",
                    0.0,
                )
            ),

            "tls_version": data.get(
                "tls_version",
                "TLS1.3",
            ),

            "cipher_suite_count": int(
                data.get(
                    "cipher_suite_count",
                    15,
                )
            ),

            "extension_count": int(
                data.get(
                    "extension_count",
                    12,
                )
            ),

            "alpn": data.get(
                "alpn",
                "h2",
            ),

            "sni_present": int(
                data.get(
                    "sni_present",
                    1,
                )
            ),
        }

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": (
                "One or more numeric features contain invalid values."
            ),
            "details": str(error),
        }), 400


    # Convert the request into a one-row DataFrame.
    input_df = pd.DataFrame([input_row])

    unknown_categories = {}


    # ---------------------------------------------------------
    # ENCODE CATEGORICAL FEATURES
    # ---------------------------------------------------------

    for column, encoder in encoders.items():
        if column not in input_df.columns:
            continue

        value = str(
            input_df.loc[0, column]
        )

        known_classes = [
            str(item)
            for item in encoder.classes_
        ]

        if value in known_classes:
            encoded_value = encoder.transform(
                [value]
            )[0]

        else:
            # Use a known training category when the live value
            # was not present during model training.
            fallback_value = encoder.classes_[0]

            encoded_value = encoder.transform(
                [fallback_value]
            )[0]

            unknown_categories[column] = {
                "received": value,
                "fallback_used": str(fallback_value),
            }

        input_df.loc[0, column] = encoded_value


    # ---------------------------------------------------------
    # VERIFY AND CONVERT MODEL INPUT
    # ---------------------------------------------------------

    try:
        # Match the trained feature order exactly.
        input_df = input_df[feature_columns]

        # Make sure every model feature is numeric.
        for column in feature_columns:
            input_df[column] = pd.to_numeric(
                input_df[column],
                errors="raise",
            )

        input_df = input_df.astype(float)

    except (KeyError, TypeError, ValueError) as error:
        return jsonify({
            "error": (
                "The API input features do not match the trained models."
            ),
            "details": str(error),
        }), 500


    # ---------------------------------------------------------
    # RUN GRADIENT BOOSTING SUPERVISED PREDICTION
    # ---------------------------------------------------------

    try:
        prediction_encoded = gradient_model.predict(
            input_df
        )[0]

        ml_prediction = label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        probabilities = gradient_model.predict_proba(
            input_df
        )[0]

        confidence = round(
            float(max(probabilities)),
            4,
        )

        class_probabilities = {}

        for class_index, probability in zip(
            gradient_model.classes_,
            probabilities,
        ):
            class_name = label_encoder.inverse_transform(
                [class_index]
            )[0]

            class_probabilities[str(class_name)] = round(
                float(probability),
                4,
            )

    except Exception as error:
        return jsonify({
            "error": (
                "Gradient Boosting could not complete the prediction."
            ),
            "details": str(error),
        }), 500


    # ---------------------------------------------------------
    # RUN ISOLATION FOREST UNSUPERVISED PREDICTION
    # ---------------------------------------------------------

    try:
        # Isolation Forest returns:
        #  1 = normal event
        # -1 = anomalous event
        isolation_raw_prediction = int(
            isolation_model.predict(
                input_df
            )[0]
        )

        if isolation_raw_prediction == -1:
            isolation_prediction = "anomaly"
            anomaly_detected = True

        else:
            isolation_prediction = "normal"
            anomaly_detected = False

        # Positive values normally indicate a more normal event.
        # Negative values normally indicate a more anomalous event.
        isolation_decision_score = round(
            float(
                isolation_model.decision_function(
                    input_df
                )[0]
            ),
            6,
        )

    except Exception as error:
        return jsonify({
            "error": (
                "Isolation Forest could not complete the prediction."
            ),
            "details": str(error),
        }), 500


    # ---------------------------------------------------------
    # RETURN BOTH MODEL RESULTS
    # ---------------------------------------------------------

    response = {
        # Supervised Gradient Boosting result.
        "ml_prediction": str(ml_prediction),
        "confidence": confidence,
        "class_probabilities": class_probabilities,

        # Unsupervised Isolation Forest result.
        "isolation_prediction": isolation_prediction,
        "anomaly_detected": anomaly_detected,
        "isolation_raw_prediction": isolation_raw_prediction,
        "isolation_decision_score": isolation_decision_score,

        # Model information.
        "models": {
            "supervised": gradient_model_name,
            "unsupervised": isolation_model_name,
        },

        # Input information.
        "features_used": input_row,
        "feature_count": len(feature_columns),
        "api_version": "dual-model-14-feature-api",

        "privacy_note": (
            "Only minimized telemetry is used. No passwords, cookies, "
            "tokens, names, emails, private content, exact location, "
            "or raw IP addresses are collected."
        ),
    }

    if unknown_categories:
        response[
            "unknown_category_fallbacks"
        ] = unknown_categories

    return jsonify(
        response
    ), 200


# ---------------------------------------------------------
# LOCAL DEVELOPMENT
# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(
        os.environ.get(
            "PORT",
            8000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )