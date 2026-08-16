from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# PROJECT SETTINGS
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic_train.csv"
)

FINAL_TEST_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic_test.csv"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "ml_api"
    / "models"
)

EVALUATION_FILE = (
    PROJECT_ROOT
    / "report_evaluation"
    / "results"
    / "isolation_forest_final_evaluation.json"
)


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

BENIGN_LABELS = {
    "human",
    "good_bot",
}

SUSPICIOUS_LABELS = {
    "bad_bot",
    "scanner",
}

EXPECTED_LABELS = (
    BENIGN_LABELS
    | SUSPICIOUS_LABELS
)

# Candidate benign false-positive operating points
CANDIDATE_FALSE_POSITIVE_RATES = [
    0.025,
    0.05,
    0.075,
    0.10,
]

MAX_VALIDATION_FALSE_POSITIVE_RATE = 0.10


# -----------------------------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------------------------

def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def create_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "one_hot",
                make_one_hot_encoder(),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------

def validate_dataset(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: "
            + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            f"{dataset_name} is empty."
        )

    labels = set(
        df[TARGET]
        .fillna("unknown")
        .astype(str)
        .unique()
    )

    unexpected_labels = (
        labels - EXPECTED_LABELS
    )

    if unexpected_labels:
        raise ValueError(
            f"{dataset_name} contains unexpected labels: "
            + ", ".join(
                sorted(unexpected_labels)
            )
        )


def prepare_feature_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:

    prepared = df[FEATURES].copy()

    for column in NUMERIC_FEATURES:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    for column in CATEGORICAL_FEATURES:
        prepared[column] = (
            prepared[column]
            .fillna("unknown")
            .astype(str)
        )

    return prepared


# -----------------------------------------------------------------------------
# ANOMALY METRICS
# -----------------------------------------------------------------------------

def labels_from_scores(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:

    return (
        scores < threshold
    ).astype(int)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    normal_total = tn + fp
    anomaly_total = tp + fn

    false_positive_rate = (
        fp / normal_total
        if normal_total
        else 0.0
    )

    false_negative_rate = (
        fn / anomaly_total
        if anomaly_total
        else 0.0
    )

    return {
        "accuracy": round(
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            4,
        ),
        "balanced_accuracy": round(
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            4,
        ),
        "false_positive_rate": round(
            float(false_positive_rate),
            4,
        ),
        "false_negative_rate": round(
            float(false_negative_rate),
            4,
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# -----------------------------------------------------------------------------
# MODEL OUTPUT
# -----------------------------------------------------------------------------

def save_model_package(
    package: dict,
    filename: str,
) -> Path:

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        MODEL_DIRECTORY
        / filename
    )

    with output_file.open(
        "wb"
    ) as file:
        pickle.dump(
            package,
            file,
        )

    return output_file


# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------

def main() -> None:

    development_df = pd.read_csv(
        TRAIN_DATA_FILE
    )

    final_test_df = pd.read_csv(
        FINAL_TEST_DATA_FILE
    )

    validate_dataset(
        development_df,
        FEATURES + [TARGET],
        "Development dataset",
    )

    validate_dataset(
        final_test_df,
        FEATURES
        + [TARGET, ANOMALY_TARGET],
        "Final holdout dataset",
    )

    X_development = (
        prepare_feature_frame(
            development_df
        )
    )

    X_final_test = (
        prepare_feature_frame(
            final_test_df
        )
    )

    y_development = (
        development_df[TARGET]
        .fillna("unknown")
        .astype(str)
    )

    y_final_labels = (
        final_test_df[TARGET]
        .fillna("unknown")
        .astype(str)
    )

    y_final_anomaly = (
        pd.to_numeric(
            final_test_df[
                ANOMALY_TARGET
            ],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_development,
    )

    # Benign-only model fitting
    benign_training_mask = (
        y_train.isin(
            BENIGN_LABELS
        )
    )

    X_benign_train = (
        X_train.loc[
            benign_training_mask
        ]
    )

    if X_benign_train.empty:
        raise ValueError(
            "No benign training records found."
        )

    preprocessor = (
        create_preprocessor()
    )

    X_isolation_train = (
        preprocessor.fit_transform(
            X_benign_train
        )
    )

    X_isolation_validation = (
        preprocessor.transform(
            X_validation
        )
    )

    X_isolation_final = (
        preprocessor.transform(
            X_final_test
        )
    )

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_isolation_train
    )

    # Validation anomaly proxy
    y_validation_anomaly = (
        y_validation
        .isin(SUSPICIOUS_LABELS)
        .astype(int)
        .to_numpy()
    )

    validation_scores = (
        model.decision_function(
            X_isolation_validation
        )
    )

    benign_validation_scores = (
        validation_scores[
            y_validation_anomaly == 0
        ]
    )

    candidate_results = []

    for target_fpr in (
        CANDIDATE_FALSE_POSITIVE_RATES
    ):

        threshold = float(
            np.quantile(
                benign_validation_scores,
                target_fpr,
            )
        )

        predictions = (
            labels_from_scores(
                validation_scores,
                threshold,
            )
        )

        metrics = (
            calculate_metrics(
                y_validation_anomaly,
                predictions,
            )
        )

        candidate_results.append(
            {
                "target_false_positive_rate":
                    target_fpr,
                "threshold":
                    threshold,
                **metrics,
            }
        )

    valid_candidates = [
        result
        for result in candidate_results
        if result[
            "false_positive_rate"
        ]
        <= MAX_VALIDATION_FALSE_POSITIVE_RATE
    ]

    if not valid_candidates:
        raise RuntimeError(
            "No threshold candidate met "
            "the validation FPR limit."
        )

    selected_candidate = max(
        valid_candidates,
        key=lambda result: (
            result["f1"],
            result["balanced_accuracy"],
            result["recall"],
        ),
    )

    selected_threshold = float(
        selected_candidate[
            "threshold"
        ]
    )

    # Final holdout evaluation
    final_scores = (
        model.decision_function(
            X_isolation_final
        )
    )

    final_predictions = (
        labels_from_scores(
            final_scores,
            selected_threshold,
        )
    )

    final_metrics = (
        calculate_metrics(
            y_final_anomaly,
            final_predictions,
        )
    )

    print(
        "\n-----------------------------------"
    )
    print(
        "ISOLATION FOREST THRESHOLD SEARCH"
    )
    print(
        "-----------------------------------"
    )

    for result in candidate_results:

        print(
            "\nTarget benign FPR:",
            result[
                "target_false_positive_rate"
            ],
        )

        print(
            "Threshold:",
            round(
                result["threshold"],
                6,
            ),
        )

        print(
            "Validation Precision:",
            result["precision"],
        )

        print(
            "Validation Recall:",
            result["recall"],
        )

        print(
            "Validation F1:",
            result["f1"],
        )

        print(
            "Validation FPR:",
            result[
                "false_positive_rate"
            ],
        )

    print(
        "\n-----------------------------------"
    )
    print(
        "SELECTED VALIDATION THRESHOLD"
    )
    print(
        "-----------------------------------"
    )

    print(
        "Selected target FPR:",
        selected_candidate[
            "target_false_positive_rate"
        ],
    )

    print(
        "Selected threshold:",
        round(
            selected_threshold,
            6,
        ),
    )

    print(
        "Validation F1:",
        selected_candidate["f1"],
    )

    print(
        "Validation recall:",
        selected_candidate[
            "recall"
        ],
    )

    print(
        "Validation FPR:",
        selected_candidate[
            "false_positive_rate"
        ],
    )

    print(
        "\n-----------------------------------"
    )
    print(
        "FINAL HOLDOUT RESULTS"
    )
    print(
        "-----------------------------------"
    )

    print(
        "Accuracy:",
        final_metrics["accuracy"],
    )

    print(
        "Balanced Accuracy:",
        final_metrics[
            "balanced_accuracy"
        ],
    )

    print(
        "Precision:",
        final_metrics[
            "precision"
        ],
    )

    print(
        "Recall:",
        final_metrics[
            "recall"
        ],
    )

    print(
        "F1:",
        final_metrics["f1"],
    )

    print(
        "False Positive Rate:",
        final_metrics[
            "false_positive_rate"
        ],
    )

    print(
        "False Negative Rate:",
        final_metrics[
            "false_negative_rate"
        ],
    )

    print(
        "Confusion Matrix:",
        final_metrics[
            "confusion_matrix"
        ],
    )

    isolation_package = {
        "model_name":
            "Isolation Forest anomaly detector",
        "model":
            model,
        "preprocessor":
            preprocessor,
        "feature_columns":
            FEATURES,
        "categorical_features":
            CATEGORICAL_FEATURES,
        "numeric_features":
            NUMERIC_FEATURES,
        "normal_training_labels":
            sorted(BENIGN_LABELS),
        "anomaly_threshold":
            selected_threshold,
        "threshold_method":
            "internal_validation_threshold_search",
        "threshold_candidates":
            CANDIDATE_FALSE_POSITIVE_RATES,
        "selected_target_false_positive_rate":
            selected_candidate[
                "target_false_positive_rate"
            ],
        "validation_metrics":
            selected_candidate,
        "training_random_state":
            RANDOM_STATE,
        "scikit_learn_version":
            sklearn.__version__,
    }

    model_file = (
        save_model_package(
            isolation_package,
            "isolation_forest_model_final.pkl",
        )
    )

    evaluation = {
        "model":
            "Isolation Forest anomaly detector",
        "development_dataset":
            TRAIN_DATA_FILE.name,
        "final_holdout_dataset":
            FINAL_TEST_DATA_FILE.name,
        "feature_count":
            len(FEATURES),
        "training_records":
            int(
                len(
                    X_benign_train
                )
            ),
        "labels_passed_to_fit":
            False,
        "normal_training_labels":
            sorted(BENIGN_LABELS),
        "validation_proxy_anomaly_labels":
            sorted(SUSPICIOUS_LABELS),
        "threshold_candidates":
            candidate_results,
        "selected_threshold":
            selected_threshold,
        "selected_validation_result":
            selected_candidate,
        "final_holdout":
            final_metrics,
        "scikit_learn_version":
            sklearn.__version__,
    }

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

    print(
        "\nFinal model saved:",
        model_file,
    )

    print(
        "Evaluation saved:",
        EVALUATION_FILE,
    )


if __name__ == "__main__":
    main()