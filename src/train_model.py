import json
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

# Locate the main Salience-demo project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "synthetic_telemetry.csv"

MODEL_DIRECTORIES = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "ml_api" / "models",
]

EVALUATION_FILE = (
    PROJECT_ROOT
    / "dashboard"
    / "dashboard"
    / "model_evaluation.json"
)


# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------

FEATURES = [
    "page_path",
    "interaction_type",
    "scroll_depth_category",
    "request_interval_seconds",
    "user_agent_category",
    "has_favicon_request",
    "requested_robots_txt",
    "pages_per_session",
    "error_rate",
    "tls_version",
    "cipher_suite_count",
    "extension_count",
    "alpn",
    "sni_present",
]

TARGET = "label"


# ---------------------------------------------------------
# FEATURE ENCODING
# ---------------------------------------------------------

def encode_features(df: pd.DataFrame):
    """
    Convert categorical text features into numeric values.

    Gradient Boosting and Isolation Forest require numeric inputs.
    A separate LabelEncoder is saved for each categorical feature
    so the API can apply the same transformation later.
    """

    encoded = df[FEATURES].copy()
    encoders = {}

    for column in encoded.columns:

        # Encode any feature that is not numeric.
        if not pd.api.types.is_numeric_dtype(encoded[column]):
            encoder = LabelEncoder()

            encoded[column] = (
                encoded[column]
                .fillna("unknown")
                .astype(str)
            )

            encoded[column] = encoder.fit_transform(
                encoded[column]
            )

            encoders[column] = encoder

        else:
            # Convert numeric features safely.
            encoded[column] = pd.to_numeric(
                encoded[column],
                errors="coerce",
            )

            # Replace missing numeric values with zero.
            encoded[column] = encoded[column].fillna(0)

    return encoded, encoders


# ---------------------------------------------------------
# FALSE-POSITIVE RATE
# ---------------------------------------------------------

def false_positive_rate(y_true, y_pred, label_encoder):
    """
    Calculate how often normal traffic is incorrectly classified
    as suspicious traffic.
    """

    suspicious_labels = {
        "bad_bot",
        "scanner",
    }

    true_labels = label_encoder.inverse_transform(y_true)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    false_positives = 0
    true_negatives = 0

    for true_label, predicted_label in zip(
        true_labels,
        predicted_labels,
    ):
        true_is_suspicious = true_label in suspicious_labels
        predicted_is_suspicious = (
            predicted_label in suspicious_labels
        )

        if (
            not true_is_suspicious
            and predicted_is_suspicious
        ):
            false_positives += 1

        if (
            not true_is_suspicious
            and not predicted_is_suspicious
        ):
            true_negatives += 1

    denominator = false_positives + true_negatives

    if denominator == 0:
        return 0.0

    return false_positives / denominator


# ---------------------------------------------------------
# MAIN TRAINING PROCESS
# ---------------------------------------------------------

def main():

    # Confirm that the telemetry dataset exists.
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset was not found at: {DATA_FILE}"
        )

    # Load the synthetic telemetry dataset.
    df = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Dataset path:", DATA_FILE)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # Confirm that all required columns exist.
    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing these required columns: "
            + ", ".join(missing_columns)
        )

    # Encode the 14 model features.
    X, encoders = encode_features(df)

    # Encode the prediction labels.
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        df[TARGET]
        .fillna("unknown")
        .astype(str)
    )

    print("\nFeatures used:", len(FEATURES))
    print("Feature names:", FEATURES)
    print("Classes:", list(label_encoder.classes_))

    # Split the dataset into training and testing portions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))


    # ---------------------------------------------------------
    # GRADIENT BOOSTING MODEL
    # ---------------------------------------------------------

    gradient_model = GradientBoostingClassifier(
        random_state=42
    )

    gradient_model.fit(
        X_train,
        y_train,
    )

    print("\nGradient Boosting training completed.")


    # ---------------------------------------------------------
    # ISOLATION FOREST MODEL
    # ---------------------------------------------------------

    isolation_model = IsolationForest(
        n_estimators=200,
        contamination=0.18,
        random_state=42,
    )

    isolation_model.fit(X_train)

    print("Isolation Forest training completed.")


    # ---------------------------------------------------------
    # GRADIENT BOOSTING EVALUATION
    # ---------------------------------------------------------

    y_pred = gradient_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    all_class_indexes = list(
        range(len(label_encoder.classes_))
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=all_class_indexes,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=all_class_indexes,
    )

    readable_report = classification_report(
        y_test,
        y_pred,
        labels=all_class_indexes,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    print("\nGradient Boosting evaluation complete.")
    print("Accuracy:", round(float(accuracy), 4))

    print("\nClassification Report:")
    print(readable_report)

    print("\nConfusion Matrix:")
    print(matrix)


    # ---------------------------------------------------------
    # CREATE MODEL PACKAGES
    # ---------------------------------------------------------

    gradient_package = {
        "model_name": (
            "Gradient Boosting supervised classifier"
        ),
        "model": gradient_model,
        "feature_columns": FEATURES,
        "encoders": encoders,
        "label_encoder": label_encoder,
        "scikit_learn_version": sklearn.__version__,
    }

    isolation_package = {
        "model_name": (
            "Isolation Forest unsupervised anomaly detector"
        ),
        "model": isolation_model,
        "feature_columns": FEATURES,
        "encoders": encoders,
        "scikit_learn_version": sklearn.__version__,
    }


    # ---------------------------------------------------------
    # SAVE MODEL FILES
    # ---------------------------------------------------------

    for directory in MODEL_DIRECTORIES:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        gradient_file = (
            directory
            / "gradient_boosting_model.pkl"
        )

        isolation_file = (
            directory
            / "isolation_forest_model.pkl"
        )

        compatibility_file = (
            directory
            / "bot_detection_model.pkl"
        )

        with open(gradient_file, "wb") as file:
            pickle.dump(
                gradient_package,
                file,
            )

        with open(isolation_file, "wb") as file:
            pickle.dump(
                isolation_package,
                file,
            )

        # This is an extra copy of the Gradient Boosting model.
        # It supports older API code that expects this filename.
        with open(compatibility_file, "wb") as file:
            pickle.dump(
                gradient_package,
                file,
            )


    # ---------------------------------------------------------
    # PREPARE DASHBOARD EVALUATION RESULTS
    # ---------------------------------------------------------

    class_metrics = []

    for class_name in label_encoder.classes_:
        metrics = report.get(
            class_name,
            {},
        )

        class_metrics.append({
            "class": str(class_name),
            "precision": round(
                float(metrics.get("precision", 0)),
                3,
            ),
            "recall": round(
                float(metrics.get("recall", 0)),
                3,
            ),
            "f1_score": round(
                float(metrics.get("f1-score", 0)),
                3,
            ),
        })

    evaluation = {
        "dataset": "Synthetic telemetry dataset",
        "total_records": int(len(df)),
        "training_records": int(len(X_train)),
        "test_records": int(len(y_test)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "model_family": (
            "Rule-based baseline + Isolation Forest "
            "+ Gradient Boosting"
        ),
        "deployed_model": (
            "Gradient Boosting supervised classifier"
        ),
        "scikit_learn_version": sklearn.__version__,
        "accuracy": round(
            float(accuracy),
            3,
        ),
        "weighted_precision": round(
            float(
                report["weighted avg"]["precision"]
            ),
            3,
        ),
        "weighted_recall": round(
            float(
                report["weighted avg"]["recall"]
            ),
            3,
        ),
        "weighted_f1": round(
            float(
                report["weighted avg"]["f1-score"]
            ),
            3,
        ),
        "false_positive_rate": round(
            float(
                false_positive_rate(
                    y_test,
                    y_pred,
                    label_encoder,
                )
            ),
            3,
        ),
        "class_metrics": class_metrics,
        "confusion_matrix_labels": list(
            label_encoder.classes_
        ),
        "confusion_matrix": matrix.tolist(),
    }


    # ---------------------------------------------------------
    # SAVE DASHBOARD EVALUATION
    # ---------------------------------------------------------

    EVALUATION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVALUATION_FILE.write_text(
        json.dumps(
            evaluation,
            indent=2,
        ),
        encoding="utf-8",
    )


    # ---------------------------------------------------------
    # FINAL CONFIRMATION
    # ---------------------------------------------------------

    print("\nAll model files were saved successfully.")

    print(
        "Gradient Boosting:",
        PROJECT_ROOT
        / "ml_api"
        / "models"
        / "gradient_boosting_model.pkl",
    )

    print(
        "Isolation Forest:",
        PROJECT_ROOT
        / "ml_api"
        / "models"
        / "isolation_forest_model.pkl",
    )

    print(
        "Compatibility alias:",
        PROJECT_ROOT
        / "ml_api"
        / "models"
        / "bot_detection_model.pkl",
    )

    print(
        "Dashboard evaluation:",
        EVALUATION_FILE,
    )

    print(
        "Scikit-learn version:",
        sklearn.__version__,
    )


if __name__ == "__main__":
    main()