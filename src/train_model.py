from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# -----------------------------------------------------------------------------
# PROJECT SETTINGS
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "synthetic_train.csv"
FINAL_TEST_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "synthetic_test.csv"

# Save each trained model directly in the ML API models directory.
# This keeps one authoritative copy and avoids a duplicate root-level models folder.
MODEL_DIRECTORY = PROJECT_ROOT / "ml_api" / "models"
EVALUATION_FILE = PROJECT_ROOT / "dashboard" / "dashboard" / "model_evaluation.json"

FEATURES = [
    "page_category",
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

CATEGORICAL_FEATURES = [
    "page_category",
    "interaction_type",
    "scroll_depth_category",
    "user_agent_category",
    "tls_version",
    "alpn",
]

NUMERIC_FEATURES = [
    "request_interval_seconds",
    "has_favicon_request",
    "requested_robots_txt",
    "pages_per_session",
    "error_rate",
    "cipher_suite_count",
    "extension_count",
    "sni_present",
]

TARGET = "label"
ANOMALY_TARGET = "anomaly_ground_truth"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

# Human and known-good bot traffic define expected behaviour for Isolation Forest.
BENIGN_LABELS = {"human", "good_bot"}
SUSPICIOUS_LABELS = {"bad_bot", "scanner"}
EXPECTED_LABELS = BENIGN_LABELS | SUSPICIOUS_LABELS

# The anomaly threshold is calibrated from benign validation traffic.
# A value of 0.05 allows approximately 5% of known-benign validation records
# to fall below the threshold.
ISOLATION_TARGET_FALSE_POSITIVE_RATE = 0.05


# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------

def make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense encoder while supporting common scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def create_preprocessor() -> ColumnTransformer:
    """Create preprocessing shared by both models."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# -----------------------------------------------------------------------------
# DATA CHECKS
# -----------------------------------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Check the dataset before model training starts."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(f"{dataset_name} is empty.")

    labels = set(df[TARGET].fillna("unknown").astype(str).unique())
    unexpected_labels = labels - EXPECTED_LABELS
    if unexpected_labels:
        raise ValueError(
            f"{dataset_name} contains unexpected labels: "
            + ", ".join(sorted(unexpected_labels))
        )

    for column in NUMERIC_FEATURES:
        numeric_values = pd.to_numeric(df[column], errors="coerce")
        invalid_count = int(numeric_values.isna().sum())
        if invalid_count:
            raise ValueError(
                f"{dataset_name} contains {invalid_count} invalid numeric "
                f"values in '{column}'."
            )

    request_intervals = pd.to_numeric(
        df["request_interval_seconds"], errors="coerce"
    )
    if (request_intervals <= 0).any():
        raise ValueError(
            f"{dataset_name} contains non-positive request intervals."
        )

    error_rates = pd.to_numeric(df["error_rate"], errors="coerce")
    if ((error_rates < 0) | (error_rates > 1)).any():
        raise ValueError(
            f"{dataset_name} contains error_rate values outside 0 to 1."
        )

    for column in ["has_favicon_request", "requested_robots_txt", "sni_present"]:
        values = set(pd.to_numeric(df[column], errors="raise").astype(int).unique())
        if not values.issubset({0, 1}):
            raise ValueError(
                f"{dataset_name} contains invalid binary values in '{column}'."
            )


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only model inputs and apply the expected data types."""
    prepared = df[FEATURES].copy()

    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        prepared[column] = prepared[column].fillna("unknown").astype(str)

    return prepared


# -----------------------------------------------------------------------------
# GRADIENT BOOSTING METRICS
# -----------------------------------------------------------------------------

def supervised_false_positive_rate(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> float:
    """Measure benign traffic incorrectly classified as suspicious."""
    benign_mask = np.isin(true_labels, list(BENIGN_LABELS))
    suspicious_predictions = np.isin(
        predicted_labels, list(SUSPICIOUS_LABELS)
    )

    benign_total = int(np.sum(benign_mask))
    if benign_total == 0:
        return 0.0

    false_positives = int(np.sum(benign_mask & suspicious_predictions))
    return false_positives / benign_total


def build_supervised_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[str, Any]:
    """Create metrics for the four-class Gradient Boosting model."""
    class_indexes = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_true,
        y_pred,
        labels=class_indexes,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_indexes)

    true_text = label_encoder.inverse_transform(y_true)
    predicted_text = label_encoder.inverse_transform(y_pred)

    class_metrics = []
    for class_name in label_encoder.classes_:
        values = report.get(class_name, {})
        class_metrics.append(
            {
                "class": str(class_name),
                "precision": round(float(values.get("precision", 0)), 4),
                "recall": round(float(values.get("recall", 0)), 4),
                "f1_score": round(float(values.get("f1-score", 0)), 4),
                "support": int(values.get("support", 0)),
            }
        )

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)), 4
        ),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_precision": round(
            float(report["weighted avg"]["precision"]), 4
        ),
        "weighted_recall": round(float(report["weighted avg"]["recall"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "false_positive_rate": round(
            supervised_false_positive_rate(true_text, predicted_text), 4
        ),
        "class_metrics": class_metrics,
        "confusion_matrix_labels": [
            str(label) for label in label_encoder.classes_
        ],
        "confusion_matrix": matrix.tolist(),
    }


# -----------------------------------------------------------------------------
# ISOLATION FOREST METRICS
# -----------------------------------------------------------------------------

def anomaly_labels_from_scores(
    anomaly_scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert anomaly scores into 0=normal and 1=anomaly."""
    return (anomaly_scores < threshold).astype(int)


def build_isolation_metrics(
    y_true_anomaly: np.ndarray,
    y_pred_anomaly: np.ndarray,
    anomaly_scores: np.ndarray,
    traffic_labels: pd.Series,
) -> dict[str, Any]:
    """Create binary anomaly-detection metrics."""
    matrix = confusion_matrix(
        y_true_anomaly,
        y_pred_anomaly,
        labels=[0, 1],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    normal_total = true_negative + false_positive
    anomaly_total = true_positive + false_negative

    false_positive_rate = (
        false_positive / normal_total if normal_total else 0.0
    )
    false_negative_rate = (
        false_negative / anomaly_total if anomaly_total else 0.0
    )

    evaluation_frame = pd.DataFrame(
        {
            "label": traffic_labels.astype(str).reset_index(drop=True),
            "predicted_anomaly": y_pred_anomaly,
            "ground_truth_anomaly": y_true_anomaly,
        }
    )

    per_class_anomaly_rates = []
    for class_name in sorted(evaluation_frame["label"].unique()):
        subset = evaluation_frame[evaluation_frame["label"] == class_name]
        per_class_anomaly_rates.append(
            {
                "class": class_name,
                "records": int(len(subset)),
                "predicted_anomaly_rate": round(
                    float(subset["predicted_anomaly"].mean()), 4
                ),
                "ground_truth_anomaly_rate": round(
                    float(subset["ground_truth_anomaly"].mean()), 4
                ),
            }
        )

    return {
        "accuracy": round(
            float(accuracy_score(y_true_anomaly, y_pred_anomaly)), 4
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true_anomaly, y_pred_anomaly)), 4
        ),
        "anomaly_precision": round(
            float(
                precision_score(
                    y_true_anomaly,
                    y_pred_anomaly,
                    zero_division=0,
                )
            ),
            4,
        ),
        "anomaly_recall": round(
            float(
                recall_score(
                    y_true_anomaly,
                    y_pred_anomaly,
                    zero_division=0,
                )
            ),
            4,
        ),
        "anomaly_f1": round(
            float(
                f1_score(
                    y_true_anomaly,
                    y_pred_anomaly,
                    zero_division=0,
                )
            ),
            4,
        ),
        "false_positive_rate": round(float(false_positive_rate), 4),
        "false_negative_rate": round(float(false_negative_rate), 4),
        "predicted_anomaly_rate": round(float(np.mean(y_pred_anomaly)), 4),
        "ground_truth_anomaly_rate": round(float(np.mean(y_true_anomaly)), 4),
        "anomaly_score_summary": {
            "minimum": round(float(np.min(anomaly_scores)), 6),
            "mean": round(float(np.mean(anomaly_scores)), 6),
            "maximum": round(float(np.max(anomaly_scores)), 6),
        },
        "confusion_matrix_labels": ["normal", "anomaly"],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_counts": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
        "per_class_anomaly_rates": per_class_anomaly_rates,
    }


# -----------------------------------------------------------------------------
# MODEL OUTPUT
# -----------------------------------------------------------------------------

def save_model_package(package: dict[str, Any], filename: str) -> Path:
    """Save one model package in the ML API models directory."""
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_file = MODEL_DIRECTORY / filename

    with output_file.open("wb") as file:
        pickle.dump(package, file)

    return output_file


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

def main() -> None:
    for data_file in [TRAIN_DATA_FILE, FINAL_TEST_DATA_FILE]:
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset was not found at: {data_file}")

    development_df = pd.read_csv(TRAIN_DATA_FILE)
    final_test_df = pd.read_csv(FINAL_TEST_DATA_FILE)

    validate_dataset(
        development_df,
        FEATURES + [TARGET],
        "Development/training dataset",
    )
    validate_dataset(
        final_test_df,
        FEATURES + [TARGET, ANOMALY_TARGET],
        "Final holdout dataset",
    )

    anomaly_values = set(
        pd.to_numeric(
            final_test_df[ANOMALY_TARGET], errors="raise"
        ).astype(int).unique()
    )
    if not anomaly_values.issubset({0, 1}):
        raise ValueError(
            "Final holdout contains invalid anomaly_ground_truth values."
        )

    print("Development dataset:", TRAIN_DATA_FILE)
    print("Development rows:", len(development_df))
    print(
        "Development class counts:",
        dict(Counter(development_df[TARGET].astype(str))),
    )
    print("\nFinal holdout dataset:", FINAL_TEST_DATA_FILE)
    print("Final holdout rows:", len(final_test_df))
    print(
        "Final holdout class counts:",
        dict(Counter(final_test_df[TARGET].astype(str))),
    )
    print("\nFeatures used:", len(FEATURES))
    print("Feature names:", FEATURES)

    X_development = prepare_feature_frame(development_df)
    X_final_test = prepare_feature_frame(final_test_df)

    y_development_labels = (
        development_df[TARGET].fillna("unknown").astype(str)
    )
    y_final_test_labels = (
        final_test_df[TARGET].fillna("unknown").astype(str)
    )
    y_final_anomaly = pd.to_numeric(
        final_test_df[ANOMALY_TARGET], errors="raise"
    ).astype(int).to_numpy()

    (
        X_train,
        X_validation,
        y_train_labels,
        y_validation_labels,
    ) = train_test_split(
        X_development,
        y_development_labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development_labels,
    )

    print("\nDevelopment training records:", len(X_train))
    print("Internal validation records:", len(X_validation))

    # -------------------------------------------------------------------------
    # Gradient Boosting classifies human, good_bot, bad_bot, and scanner.
    # -------------------------------------------------------------------------

    gradient_preprocessor = create_preprocessor()
    X_train_gradient = gradient_preprocessor.fit_transform(X_train)
    X_validation_gradient = gradient_preprocessor.transform(X_validation)
    X_final_gradient = gradient_preprocessor.transform(X_final_test)

    gradient_label_encoder = LabelEncoder()
    y_train_gradient = gradient_label_encoder.fit_transform(y_train_labels)
    y_validation_gradient = gradient_label_encoder.transform(
        y_validation_labels
    )
    y_final_gradient = gradient_label_encoder.transform(y_final_test_labels)

    gradient_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gradient_model.fit(X_train_gradient, y_train_gradient)

    validation_predictions = gradient_model.predict(X_validation_gradient)
    final_gradient_predictions = gradient_model.predict(X_final_gradient)

    validation_metrics = build_supervised_metrics(
        y_validation_gradient,
        validation_predictions,
        gradient_label_encoder,
    )
    final_gradient_metrics = build_supervised_metrics(
        y_final_gradient,
        final_gradient_predictions,
        gradient_label_encoder,
    )

    print("\nGradient Boosting training completed.")
    print("Internal validation accuracy:", validation_metrics["accuracy"])
    print("Final holdout accuracy:", final_gradient_metrics["accuracy"])

    # -------------------------------------------------------------------------
    # Isolation Forest learns expected traffic from human and good-bot records.
    # Bad-bot and scanner records are excluded from model fitting.
    # -------------------------------------------------------------------------

    benign_training_mask = y_train_labels.isin(BENIGN_LABELS)
    benign_validation_mask = y_validation_labels.isin(BENIGN_LABELS)

    X_isolation_benign_train = X_train.loc[benign_training_mask]
    X_isolation_benign_validation = X_validation.loc[
        benign_validation_mask
    ]

    if X_isolation_benign_train.empty:
        raise ValueError(
            "No benign records were available for Isolation Forest training."
        )

    if X_isolation_benign_validation.empty:
        raise ValueError(
            "No benign records were available for threshold calibration."
        )

    isolation_preprocessor = create_preprocessor()
    X_isolation_train = isolation_preprocessor.fit_transform(
        X_isolation_benign_train
    )
    X_isolation_validation = isolation_preprocessor.transform(
        X_isolation_benign_validation
    )
    X_isolation_final = isolation_preprocessor.transform(X_final_test)

    isolation_model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Isolation Forest receives feature values only; no class labels are passed.
    isolation_model.fit(X_isolation_train)

    # Lower decision_function values represent more unusual records.
    benign_validation_scores = isolation_model.decision_function(
        X_isolation_validation
    )
    isolation_threshold = float(
        np.quantile(
            benign_validation_scores,
            ISOLATION_TARGET_FALSE_POSITIVE_RATE,
        )
    )

    final_isolation_scores = isolation_model.decision_function(
        X_isolation_final
    )
    final_isolation_predictions = anomaly_labels_from_scores(
        final_isolation_scores,
        isolation_threshold,
    )

    isolation_metrics = build_isolation_metrics(
        y_true_anomaly=y_final_anomaly,
        y_pred_anomaly=final_isolation_predictions,
        anomaly_scores=final_isolation_scores,
        traffic_labels=y_final_test_labels,
    )

    print("\nIsolation Forest training completed.")
    print("Benign training records:", len(X_isolation_benign_train))
    print(
        "Benign threshold-calibration records:",
        len(X_isolation_benign_validation),
    )
    print("Calibrated anomaly threshold:", round(isolation_threshold, 6))
    print("Final anomaly F1:", isolation_metrics["anomaly_f1"])
    print(
        "Predicted anomaly rate:",
        isolation_metrics["predicted_anomaly_rate"],
    )

    # -------------------------------------------------------------------------
    # Model packages include preprocessing and metadata needed by the API.
    # -------------------------------------------------------------------------

    gradient_package = {
        "model_name": "Gradient Boosting supervised classifier",
        "model": gradient_model,
        "preprocessor": gradient_preprocessor,
        "feature_columns": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "label_encoder": gradient_label_encoder,
        "training_random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
    }

    isolation_package = {
        "model_name": "Isolation Forest anomaly detector",
        "model": isolation_model,
        "preprocessor": isolation_preprocessor,
        "feature_columns": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "normal_training_labels": sorted(BENIGN_LABELS),
        "anomaly_threshold": isolation_threshold,
        "target_benign_false_positive_rate": (
            ISOLATION_TARGET_FALSE_POSITIVE_RATE
        ),
        "training_random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
    }

    gradient_model_file = save_model_package(
        gradient_package,
        "gradient_boosting_model.pkl",
    )
    isolation_model_file = save_model_package(
        isolation_package,
        "isolation_forest_model.pkl",
    )

    # -------------------------------------------------------------------------
    # The dashboard file contains evaluation results, not trained model inputs.
    # -------------------------------------------------------------------------

    evaluation = {
        "dataset": "Realistic synthetic telemetry datasets",
        "development_dataset_file": TRAIN_DATA_FILE.name,
        "final_holdout_file": FINAL_TEST_DATA_FILE.name,
        "development_records": int(len(development_df)),
        "development_training_records": int(len(X_train)),
        "internal_validation_records": int(len(X_validation)),
        "final_holdout_records": int(len(final_test_df)),
        "development_class_counts": dict(
            Counter(development_df[TARGET].astype(str))
        ),
        "final_holdout_class_counts": dict(
            Counter(final_test_df[TARGET].astype(str))
        ),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "random_state": RANDOM_STATE,
        "scikit_learn_version": sklearn.__version__,
        "gradient_boosting": {
            "model_name": "Gradient Boosting supervised classifier",
            "internal_validation": validation_metrics,
            "final_holdout": final_gradient_metrics,
        },
        "isolation_forest": {
            "model_name": "Isolation Forest anomaly detector",
            "training_records": int(len(X_isolation_benign_train)),
            "threshold_calibration_records": int(
                len(X_isolation_benign_validation)
            ),
            "training_labels_used_for_selection": sorted(BENIGN_LABELS),
            "labels_passed_to_fit": False,
            "threshold_method": "benign_validation_quantile",
            "target_benign_false_positive_rate": (
                ISOLATION_TARGET_FALSE_POSITIVE_RATE
            ),
            "anomaly_threshold": isolation_threshold,
            "final_holdout": isolation_metrics,
        },
        # These fields remain available for the current dashboard layout.
        "total_records": int(len(final_test_df)),
        "training_records": int(len(X_train)),
        "test_records": int(len(final_test_df)),
        "model_family": "Gradient Boosting + Isolation Forest",
        "deployed_model": "Gradient Boosting supervised classifier",
        "accuracy": final_gradient_metrics["accuracy"],
        "weighted_precision": final_gradient_metrics["weighted_precision"],
        "weighted_recall": final_gradient_metrics["weighted_recall"],
        "weighted_f1": final_gradient_metrics["weighted_f1"],
        "false_positive_rate": final_gradient_metrics[
            "false_positive_rate"
        ],
        "class_metrics": final_gradient_metrics["class_metrics"],
        "confusion_matrix_labels": final_gradient_metrics[
            "confusion_matrix_labels"
        ],
        "confusion_matrix": final_gradient_metrics["confusion_matrix"],
    }

    EVALUATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_FILE.write_text(
        json.dumps(evaluation, indent=2),
        encoding="utf-8",
    )

    print("\nAll model files were saved successfully.")
    print("Gradient Boosting:", gradient_model_file)
    print("Isolation Forest:", isolation_model_file)
    print("Dashboard evaluation:", EVALUATION_FILE)
    print("Scikit-learn version:", sklearn.__version__)


if __name__ == "__main__":
    main()