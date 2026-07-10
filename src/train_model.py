import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_FILE = Path("data/raw/synthetic_telemetry.csv")
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


def encode_features(df: pd.DataFrame):
    encoded = df[FEATURES].copy()
    encoders = {}

    for column in encoded.columns:
        if encoded[column].dtype == "object":
            encoder = LabelEncoder()
            encoded[column] = encoder.fit_transform(encoded[column].astype(str))
            encoders[column] = encoder

    return encoded, encoders


def false_positive_rate(y_true, y_pred, label_encoder):
    suspicious = {"bad_bot", "scanner"}
    true_labels = label_encoder.inverse_transform(y_true)
    pred_labels = label_encoder.inverse_transform(y_pred)

    fp = 0
    tn = 0
    for true_label, pred_label in zip(true_labels, pred_labels):
        true_is_suspicious = true_label in suspicious
        pred_is_suspicious = pred_label in suspicious
        if not true_is_suspicious and pred_is_suspicious:
            fp += 1
        if not true_is_suspicious and not pred_is_suspicious:
            tn += 1

    return fp / (fp + tn) if (fp + tn) else 0.0


def main():
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded successfully.")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    X, encoders = encode_features(df)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    gradient_model = GradientBoostingClassifier(random_state=42)
    gradient_model.fit(X_train, y_train)

    isolation_model = IsolationForest(
        n_estimators=200,
        contamination=0.18,
        random_state=42,
    )
    isolation_model.fit(X_train)

    y_pred = gradient_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred)

    print("\nGradient Boosting model training complete.")
    print("Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    print("\nConfusion Matrix:")
    print(matrix)

    model_dirs = [Path("models"), Path("ml_api/models")]
    for directory in model_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    gradient_package = {
        "model_name": "Gradient Boosting supervised classifier",
        "model": gradient_model,
        "feature_columns": FEATURES,
        "encoders": encoders,
        "label_encoder": label_encoder,
    }
    isolation_package = {
        "model_name": "Isolation Forest unsupervised anomaly detector",
        "model": isolation_model,
        "feature_columns": FEATURES,
        "encoders": encoders,
    }

    for directory in model_dirs:
        with open(directory / "gradient_boosting_model.pkl", "wb") as file:
            pickle.dump(gradient_package, file)
        with open(directory / "isolation_forest_model.pkl", "wb") as file:
            pickle.dump(isolation_package, file)
        # Compatibility alias for older deployment startup settings.
        with open(directory / "bot_detection_model.pkl", "wb") as file:
            pickle.dump(gradient_package, file)

    class_metrics = []
    for class_name in label_encoder.classes_:
        metrics = report.get(class_name, {})
        class_metrics.append({
            "class": class_name,
            "precision": round(float(metrics.get("precision", 0)), 3),
            "recall": round(float(metrics.get("recall", 0)), 3),
            "f1_score": round(float(metrics.get("f1-score", 0)), 3),
        })

    evaluation = {
        "dataset": "Synthetic telemetry dataset",
        "test_records": int(len(y_test)),
        "model_family": "Rule-based baseline + Isolation Forest + Gradient Boosting",
        "accuracy": round(float(accuracy), 3),
        "weighted_precision": round(float(report["weighted avg"]["precision"]), 3),
        "weighted_recall": round(float(report["weighted avg"]["recall"]), 3),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 3),
        "false_positive_rate": round(float(false_positive_rate(y_test, y_pred, label_encoder)), 3),
        "class_metrics": class_metrics,
        "confusion_matrix_labels": list(label_encoder.classes_),
        "confusion_matrix": matrix.tolist(),
    }

    eval_path = Path("dashboard/dashboard/model_evaluation.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    print("\nSaved supervised model to models/gradient_boosting_model.pkl and ml_api/models/gradient_boosting_model.pkl")
    print("Saved unsupervised model to models/isolation_forest_model.pkl and ml_api/models/isolation_forest_model.pkl")
    print("Saved compatibility alias to bot_detection_model.pkl")
    print("Saved dashboard evaluation to dashboard/dashboard/model_evaluation.json")


if __name__ == "__main__":
    main()
