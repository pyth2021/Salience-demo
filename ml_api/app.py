from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)


# -----------------------------------------------------------------------------
# MODEL LOCATIONS
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIRECTORY = PROJECT_ROOT / "models"

GRADIENT_MODEL_PATH = MODEL_DIRECTORY / "gradient_boosting_model.pkl"
ISOLATION_MODEL_PATH = MODEL_DIRECTORY / "isolation_forest_model.pkl"


# -----------------------------------------------------------------------------
# LOAD MODEL PACKAGES
# -----------------------------------------------------------------------------

def load_model_package(
    model_path: Path,
    model_description: str,
) -> dict[str, Any]:
    """Load and validate a saved model package."""

    try:
        with model_path.open("rb") as file:
            package = pickle.load(file)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{model_description} file was not found at: {model_path}"
        ) from error
    except Exception as error:
        raise RuntimeError(
            f"Unable to load {model_description}: {error}"
        ) from error

    if not isinstance(package, dict):
        raise RuntimeError(
            f"{model_description} package must be a dictionary."
        )

    return package


gradient_package = load_model_package(
    GRADIENT_MODEL_PATH,
    "Gradient Boosting model",
)

isolation_package = load_model_package(
    ISOLATION_MODEL_PATH,
    "Isolation Forest model",
)


# -----------------------------------------------------------------------------
# GRADIENT BOOSTING COMPONENTS
# -----------------------------------------------------------------------------

gradient_model = gradient_package["model"]
gradient_preprocessor = gradient_package["preprocessor"]
feature_columns = gradient_package["feature_columns"]
label_encoder = gradient_package["label_encoder"]

gradient_model_name = gradient_package.get(
    "model_name",
    "Gradient Boosting supervised classifier",
)


# -----------------------------------------------------------------------------
# ISOLATION FOREST COMPONENTS
# -----------------------------------------------------------------------------

isolation_model = isolation_package["model"]
isolation_preprocessor = isolation_package["preprocessor"]

isolation_feature_columns = isolation_package.get(
    "feature_columns",
    feature_columns,
)

# This threshold was calibrated using benign validation traffic during training.
isolation_threshold = float(
    isolation_package.get("anomaly_threshold", 0.0)
)

isolation_model_name = isolation_package.get(
    "model_name",
    "Isolation Forest anomaly detector",
)


# -----------------------------------------------------------------------------
# MODEL PACKAGE VALIDATION
# -----------------------------------------------------------------------------

if list(feature_columns) != list(isolation_feature_columns):
    raise RuntimeError(
        "Gradient Boosting and Isolation Forest feature columns do not match."
    )

if len(feature_columns) != 14:
    raise RuntimeError(
        f"The API expected 14 model features, but the model package "
        f"contains {len(feature_columns)}."
    )


# -----------------------------------------------------------------------------
# EXPECTED FEATURE CATEGORIES
# -----------------------------------------------------------------------------

PAGE_CATEGORIES = {
    "public_page",
    "account_page",
    "checkout_page",
    "crawler_file",
    "sensitive_page",
    "unknown_page",
}

INTERACTION_TYPES = {
    "page_view",
    "navigation",
    "resource_request",
    "form_request",
    "api_request",
    "automated_request",
}

SCROLL_CATEGORIES = {
    "none",
    "low",
    "medium",
    "high",
}

USER_AGENT_CATEGORIES = {
    "browser",
    "crawler",
    "script_client",
    "command_line",
    "unknown",
}

TLS_VERSIONS = {
    "TLS1.2",
    "TLS1.3",
}

ALPN_VALUES = {
    "h2",
    "http/1.1",
}


# -----------------------------------------------------------------------------
# INPUT NORMALIZATION
# -----------------------------------------------------------------------------

def normalize_text(value: Any, default: str) -> str:
    """Convert an input value into clean text."""

    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def normalize_category(
    value: Any,
    allowed_values: set[str],
    default: str,
) -> str:
    """Return a known category or a safe default."""

    text = normalize_text(value, default)
    return text if text in allowed_values else default


def parse_float(
    data: dict[str, Any],
    field_name: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Read and validate a floating-point feature."""

    raw_value = data.get(field_name, default)

    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a number.") from error

    if not np.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number.")

    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")

    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be no greater than {maximum}.")

    return value


def parse_integer(
    data: dict[str, Any],
    field_name: str,
    default: int,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Read and validate an integer feature."""

    raw_value = data.get(field_name, default)

    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a number.") from error

    if not np.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be a finite number.")

    if not numeric_value.is_integer():
        raise ValueError(f"{field_name} must be an integer.")

    value = int(numeric_value)

    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")

    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be no greater than {maximum}.")

    return value


def parse_binary(
    data: dict[str, Any],
    field_name: str,
    default: int,
) -> int:
    """Read a binary feature and require zero or one."""

    return parse_integer(
        data=data,
        field_name=field_name,
        default=default,
        minimum=0,
        maximum=1,
    )


# -----------------------------------------------------------------------------
# BUILD MODEL INPUT
# -----------------------------------------------------------------------------

def build_input_row(data: dict[str, Any]) -> dict[str, Any]:
    """Build one validated row containing the 14 model features."""

    return {
        "page_category": normalize_category(
            data.get("page_category"),
            PAGE_CATEGORIES,
            "unknown_page",
        ),
        "interaction_type": normalize_category(
            data.get("interaction_type"),
            INTERACTION_TYPES,
            "page_view",
        ),
        "scroll_depth_category": normalize_category(
            data.get("scroll_depth_category"),
            SCROLL_CATEGORIES,
            "medium",
        ),
        "request_interval_seconds": parse_float(
            data,
            "request_interval_seconds",
            default=10.0,
            minimum=0.001,
        ),
        "user_agent_category": normalize_category(
            data.get("user_agent_category"),
            USER_AGENT_CATEGORIES,
            "unknown",
        ),
        "has_favicon_request": parse_binary(
            data,
            "has_favicon_request",
            default=1,
        ),
        "requested_robots_txt": parse_binary(
            data,
            "requested_robots_txt",
            default=0,
        ),
        "pages_per_session": parse_integer(
            data,
            "pages_per_session",
            default=3,
            minimum=1,
        ),
        "error_rate": parse_float(
            data,
            "error_rate",
            default=0.0,
            minimum=0.0,
            maximum=1.0,
        ),
        "tls_version": normalize_category(
            data.get("tls_version"),
            TLS_VERSIONS,
            "TLS1.3",
        ),
        "cipher_suite_count": parse_integer(
            data,
            "cipher_suite_count",
            default=15,
            minimum=0,
        ),
        "extension_count": parse_integer(
            data,
            "extension_count",
            default=12,
            minimum=0,
        ),
        "alpn": normalize_category(
            data.get("alpn"),
            ALPN_VALUES,
            "h2",
        ),
        "sni_present": parse_binary(
            data,
            "sni_present",
            default=1,
        ),
    }


# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "running",
            "service": "Salience ML Bot and Anomaly Detection API",
            "api_version": "dual-model-14-feature-api-v2",
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
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "models_loaded": {
                "gradient_boosting": True,
                "isolation_forest": True,
            },
            "preprocessors_loaded": {
                "gradient_boosting": True,
                "isolation_forest": True,
            },
            "api_version": "dual-model-14-feature-api-v2",
            "models": {
                "supervised": gradient_model_name,
                "unsupervised": isolation_model_name,
            },
            "feature_count": len(feature_columns),
            "features": list(feature_columns),
            "isolation_threshold": round(isolation_threshold, 8),
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify(
            {"error": "Invalid or missing JSON body."}
        ), 400

    if not isinstance(data, dict):
        return jsonify(
            {"error": "The JSON body must be an object."}
        ), 400

    # Build the raw feature row in the same order used during training.
    try:
        input_row = build_input_row(data)
        input_df = pd.DataFrame(
            [input_row],
            columns=feature_columns,
        )
    except (TypeError, ValueError, KeyError) as error:
        return jsonify(
            {
                "error": "One or more input features contain invalid values.",
                "details": str(error),
            }
        ), 400

    # Run the four-class Gradient Boosting prediction.
    try:
        gradient_input = gradient_preprocessor.transform(input_df)

        prediction_encoded = int(
            gradient_model.predict(gradient_input)[0]
        )

        ml_prediction = label_encoder.inverse_transform(
            [prediction_encoded]
        )[0]

        probabilities = gradient_model.predict_proba(
            gradient_input
        )[0]

        confidence = round(
            float(np.max(probabilities)),
            4,
        )

        class_probabilities = {}

        for class_index, probability in zip(
            gradient_model.classes_,
            probabilities,
        ):
            class_name = label_encoder.inverse_transform(
                [int(class_index)]
            )[0]

            class_probabilities[str(class_name)] = round(
                float(probability),
                4,
            )

    except Exception as error:
        return jsonify(
            {
                "error": "Gradient Boosting could not complete the prediction.",
                "details": str(error),
            }
        ), 500

    # Run the Isolation Forest anomaly prediction.
    try:
        isolation_input = isolation_preprocessor.transform(input_df)

        isolation_decision_score = float(
            isolation_model.decision_function(isolation_input)[0]
        )

        anomaly_detected = bool(
            isolation_decision_score < isolation_threshold
        )

        isolation_prediction = (
            "anomaly"
            if anomaly_detected
            else "normal"
        )

    except Exception as error:
        return jsonify(
            {
                "error": "Isolation Forest could not complete the prediction.",
                "details": str(error),
            }
        ), 500

    response = {
        "ml_prediction": str(ml_prediction),
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "isolation_prediction": isolation_prediction,
        "anomaly_detected": anomaly_detected,
        "isolation_decision_score": round(
            isolation_decision_score,
            8,
        ),
        "isolation_threshold": round(
            isolation_threshold,
            8,
        ),
        "models": {
            "supervised": gradient_model_name,
            "unsupervised": isolation_model_name,
        },
        "features_used": input_row,
        "feature_count": len(feature_columns),
        "api_version": "dual-model-14-feature-api-v2",
        "privacy_note": (
            "Only minimized telemetry is used. No passwords, cookies, tokens, "
            "names, emails, private content, exact location, or raw IP "
            "addresses are collected."
        ),
    }

    return jsonify(response), 200


# -----------------------------------------------------------------------------
# LOCAL DEVELOPMENT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )