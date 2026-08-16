from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "synthetic_test.csv"
)

GRADIENT_MODEL_FILE = (
    PROJECT_ROOT
    / "ml_api"
    / "models"
    / "gradient_boosting_model.pkl"
)

BASELINE_ISOLATION_MODEL_FILE = (
    PROJECT_ROOT
    / "ml_api"
    / "models"
    / "isolation_forest_model.pkl"
)

FINAL_ISOLATION_MODEL_FILE = (
    PROJECT_ROOT
    / "ml_api"
    / "models"
    / "isolation_forest_model_final.pkl"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "report_evaluation"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
# ---------------------------------------------------------
# LOAD HOLDOUT DATASET
# ---------------------------------------------------------

test_df = pd.read_csv(TEST_DATA_FILE)

print("Final holdout dataset loaded.")
print("Rows:", len(test_df))
print("\nClass distribution:")
print(test_df["label"].value_counts())

# ---------------------------------------------------------
# LOAD GRADIENT MODEL
# ---------------------------------------------------------

with open(GRADIENT_MODEL_FILE, "rb") as file:
    gradient_package = pickle.load(file)

gradient_model = gradient_package["model"]
gradient_preprocessor = gradient_package["preprocessor"]
label_encoder = gradient_package["label_encoder"]
feature_columns = gradient_package["feature_columns"]

print(
    "Gradient Boosting classes:",
    list(label_encoder.classes_),
)
# ---------------------------------------------------------
# LOAD ISOLATION FOREST MODELS
# ---------------------------------------------------------

with open(
    BASELINE_ISOLATION_MODEL_FILE,
    "rb",
) as file:
    baseline_isolation_package = pickle.load(file)

with open(
    FINAL_ISOLATION_MODEL_FILE,
    "rb",
) as file:
    final_isolation_package = pickle.load(file)


baseline_isolation_model = (
    baseline_isolation_package["model"]
)

baseline_isolation_preprocessor = (
    baseline_isolation_package["preprocessor"]
)

baseline_isolation_threshold = float(
    baseline_isolation_package["anomaly_threshold"]
)


final_isolation_model = (
    final_isolation_package["model"]
)

final_isolation_preprocessor = (
    final_isolation_package["preprocessor"]
)

final_isolation_threshold = float(
    final_isolation_package["anomaly_threshold"]
)


print("\nIsolation Forest models loaded successfully.")

print(
    "Baseline threshold:",
    baseline_isolation_threshold,
)

print(
    "Final threshold:",
    final_isolation_threshold,
)
# ---------------------------------------------------------
# PREPARE THE SAME 14 FEATURES USED DURING TRAINING
# ---------------------------------------------------------

X_test = test_df[feature_columns].copy()

y_true_labels = (
    test_df["label"]
    .fillna("unknown")
    .astype(str)
)

y_true_anomaly = (
    pd.to_numeric(
        test_df["anomaly_ground_truth"],
        errors="raise",
    )
    .astype(int)
    .to_numpy()
)

print(
    "\nNumber of model features:",
    len(feature_columns),
)

print("Feature names:")
for feature in feature_columns:
    print("-", feature)

# ---------------------------------------------------------
# GRADIENT BOOSTING EVALUATION
# ---------------------------------------------------------

# Apply the same preprocessing used during training.
X_test_gradient = gradient_preprocessor.transform(X_test)

# Encode the true class labels using the saved label encoder.
y_true_gradient = label_encoder.transform(y_true_labels)

# Generate Gradient Boosting predictions.
y_pred_gradient = gradient_model.predict(X_test_gradient)

# ---------------------------------------------------------
# OVERALL METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_true_gradient,
    y_pred_gradient,
)

balanced_accuracy = balanced_accuracy_score(
    y_true_gradient,
    y_pred_gradient,
)

macro_precision = precision_score(
    y_true_gradient,
    y_pred_gradient,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_true_gradient,
    y_pred_gradient,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true_gradient,
    y_pred_gradient,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true_gradient,
    y_pred_gradient,
    average="weighted",
    zero_division=0,
)

print("\n-----------------------------------")
print("GRADIENT BOOSTING FINAL RESULTS")
print("-----------------------------------")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")

# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_true_gradient,
        y_pred_gradient,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

gradient_cm = confusion_matrix(
    y_true_gradient,
    y_pred_gradient,
)

print("\nConfusion Matrix:")
print(gradient_cm)

# ---------------------------------------------------------
# SAVE GRADIENT BOOSTING CONFUSION MATRIX
# ---------------------------------------------------------

# Cleaner display names for the final report.
class_names = [
    "Bad Bot",
    "Good Bot",
    "Human",
    "Scanner",
]

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    gradient_cm,
    cmap="Blues",
)

# Axis tick positions and labels.
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))

ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Gradient Boosting Confusion Matrix")
ax.title.set_position(
    (0.5, 1.03)
)

# Add a color scale.
fig.colorbar(
    image,
    ax=ax,
    label="Number of Records",
)

# Change text color
threshold = gradient_cm.max() / 2

for i in range(gradient_cm.shape[0]):
    for j in range(gradient_cm.shape[1]):
        ax.text(
            j,
            i,
            gradient_cm[i, j],
            ha="center",
            va="center",
            color=(
                "white"
                if gradient_cm[i, j] > threshold
                else "black"
            ),
        )

fig.tight_layout()
fig.subplots_adjust(
    bottom=0.16
)
confusion_matrix_file = (
    RESULTS_DIR
    / "gradient_boosting_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "\nGradient Boosting confusion matrix saved to:",
    confusion_matrix_file,
)
# ---------------------------------------------------------
# GRADIENT BOOSTING FEATURE IMPORTANCE
# ---------------------------------------------------------

# Get the transformed feature names after one-hot encoding.
transformed_feature_names = (
    gradient_preprocessor.get_feature_names_out()
)

feature_importance_values = (
    gradient_model.feature_importances_
)

feature_importance_df = pd.DataFrame(
    {
        "feature": transformed_feature_names,
        "importance": feature_importance_values,
    }
)

# ---------------------------------------------------------
# AGGREGATE IMPORTANCE TO ORIGINAL 14 FEATURES
# ---------------------------------------------------------

original_feature_importance = {}

for original_feature in feature_columns:

    matching_rows = feature_importance_df[
        feature_importance_df["feature"].apply(
            lambda name: (
                name == original_feature
                or name.startswith(original_feature + "_")
            )
        )
    ]

    original_feature_importance[original_feature] = (
        matching_rows["importance"].sum()
    )


aggregated_importance_df = pd.DataFrame(
    {
        "feature": list(original_feature_importance.keys()),
        "importance": list(original_feature_importance.values()),
    }
)

aggregated_importance_df = (
    aggregated_importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\nAggregated Importance for Original 14 Features:")
print(aggregated_importance_df)


# Cleaner labels for the report.
display_names = {
    "page_category": "Page Category",
    "interaction_type": "Interaction Type",
    "scroll_depth_category": "Scroll Depth Category",
    "request_interval_seconds": "Request Interval",
    "user_agent_category": "User-Agent Category",
    "has_favicon_request": "Favicon Request",
    "requested_robots_txt": "robots.txt Request",
    "pages_per_session": "Pages per Session",
    "error_rate": "Error Rate",
    "tls_version": "TLS Version",
    "cipher_suite_count": "Cipher Suite Count",
    "extension_count": "Extension Count",
    "alpn": "ALPN",
    "sni_present": "SNI Present",
}

aggregated_importance_df["display_name"] = (
    aggregated_importance_df["feature"]
    .map(display_names)
)


# ---------------------------------------------------------
# PLOT ORIGINAL 14 FEATURES
# ---------------------------------------------------------

plot_df = aggregated_importance_df.iloc[::-1]

fig, ax = plt.subplots(figsize=(9, 7))

ax.barh(
    plot_df["display_name"],
    plot_df["importance"],
)

ax.set_xlabel("Feature Importance")
ax.set_ylabel("Telemetry Feature")

ax.set_title(
    "Gradient Boosting Feature Importance"
)
ax.title.set_position(
    (0.5, 1.03)
)
fig.tight_layout()
fig.subplots_adjust(
    bottom=0.16
)
aggregated_plot_file = (
    RESULTS_DIR
    / "gradient_boosting_feature_importance_14_features.png"
)

plt.savefig(
    aggregated_plot_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "\nAggregated feature importance plot saved to:",
    aggregated_plot_file,
)
# ---------------------------------------------------------
# ISOLATION FOREST COMPARISON
# ---------------------------------------------------------

def evaluate_isolation_model(
    model,
    preprocessor,
    threshold,
):
    X_processed = preprocessor.transform(X_test)

    scores = model.decision_function(
        X_processed
    )

    predictions = (
        scores < threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true_anomaly,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true_anomaly,
        predictions,
    )

    precision = precision_score(
        y_true_anomaly,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true_anomaly,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true_anomaly,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true_anomaly,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "scores": scores,
        "predictions": predictions,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "confusion_matrix": matrix,
    }


baseline_results = evaluate_isolation_model(
    baseline_isolation_model,
    baseline_isolation_preprocessor,
    baseline_isolation_threshold,
)

final_results = evaluate_isolation_model(
    final_isolation_model,
    final_isolation_preprocessor,
    final_isolation_threshold,
)


print("\n-----------------------------------")
print("ISOLATION FOREST COMPARISON")
print("-----------------------------------")

print("\nBaseline Isolation Forest")
print(
    f"Threshold:          "
    f"{baseline_isolation_threshold:.6f}"
)
print(
    f"Accuracy:           "
    f"{baseline_results['accuracy']:.4f}"
)
print(
    f"Balanced Accuracy:  "
    f"{baseline_results['balanced_accuracy']:.4f}"
)
print(
    f"Precision:          "
    f"{baseline_results['precision']:.4f}"
)
print(
    f"Recall:             "
    f"{baseline_results['recall']:.4f}"
)
print(
    f"F1 Score:           "
    f"{baseline_results['f1']:.4f}"
)
print(
    f"False Positive Rate:"
    f"{baseline_results['false_positive_rate']:.4f}"
)
print(
    f"False Negative Rate:"
    f"{baseline_results['false_negative_rate']:.4f}"
)

print("\nFinal Isolation Forest")
print(
    f"Threshold:          "
    f"{final_isolation_threshold:.6f}"
)
print(
    f"Accuracy:           "
    f"{final_results['accuracy']:.4f}"
)
print(
    f"Balanced Accuracy:  "
    f"{final_results['balanced_accuracy']:.4f}"
)
print(
    f"Precision:          "
    f"{final_results['precision']:.4f}"
)
print(
    f"Recall:             "
    f"{final_results['recall']:.4f}"
)
print(
    f"F1 Score:           "
    f"{final_results['f1']:.4f}"
)
print(
    f"False Positive Rate:"
    f"{final_results['false_positive_rate']:.4f}"
)
print(
    f"False Negative Rate:"
    f"{final_results['false_negative_rate']:.4f}"
)
# ---------------------------------------------------------
# FINAL ISOLATION FOREST CONFUSION MATRIX
# ---------------------------------------------------------

final_isolation_cm = final_results["confusion_matrix"]

anomaly_class_names = [
    "Normal",
    "Anomaly",
]

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(
    final_isolation_cm,
    cmap="Blues",
)

ax.set_xticks(
    np.arange(len(anomaly_class_names))
)

ax.set_yticks(
    np.arange(len(anomaly_class_names))
)

ax.set_xticklabels(
    anomaly_class_names
)

ax.set_yticklabels(
    anomaly_class_names
)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_title(
    "Isolation Forest Confusion Matrix"
)
ax.title.set_position(
    (0.5, 1.03)
)
fig.colorbar(
    image,
    ax=ax,
    label="Number of Records",
)

text_threshold = (
    final_isolation_cm.max() / 2
)

for i in range(
    final_isolation_cm.shape[0]
):
    for j in range(
        final_isolation_cm.shape[1]
    ):
        ax.text(
            j,
            i,
            final_isolation_cm[i, j],
            ha="center",
            va="center",
            color=(
                "white"
                if final_isolation_cm[i, j]
                > text_threshold
                else "black"
            ),
        )

fig.tight_layout()
fig.subplots_adjust(
    bottom=0.16
)

final_isolation_cm_file = (
    RESULTS_DIR
    / "isolation_forest_final_confusion_matrix.png"
)

plt.savefig(
    final_isolation_cm_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "\nFinal Isolation Forest confusion matrix saved to:",
    final_isolation_cm_file,
)

# ---------------------------------------------------------
# ISOLATION FOREST ANOMALY SCORE DISTRIBUTION
# ---------------------------------------------------------

final_isolation_scores = final_results["scores"]

# Separate scores using the final holdout ground-truth labels
normal_scores = final_isolation_scores[
    y_true_anomaly == 0
]

anomaly_scores = final_isolation_scores[
    y_true_anomaly == 1
]

print("\nScore distribution records:")
print("Normal:", len(normal_scores))
print("Anomaly:", len(anomaly_scores))

# Create normalized score distributions
fig, ax = plt.subplots(figsize=(9, 6))

ax.hist(
    normal_scores,
    bins=50,
    alpha=0.6,
    density=True,
    label="Normal",
)

ax.hist(
    anomaly_scores,
    bins=50,
    alpha=0.6,
    density=True,
    label="Anomaly",
)

# Mark the selected final anomaly threshold
ax.axvline(
    final_isolation_threshold,
    linestyle="--",
    linewidth=2,
    label=(
        f"Threshold = "
        f"{final_isolation_threshold:.4f}"
    ),
)

ax.set_xlabel(
    "Isolation Forest Decision Score"
)
ax.title.set_position(
    (0.5, 1.03)
)
ax.set_ylabel(
    "Density"
)

ax.set_title(
    "Isolation Forest Anomaly Score Distribution"
)

ax.legend()

fig.tight_layout()
fig.subplots_adjust(
    bottom=0.16
)

score_distribution_file = (
    RESULTS_DIR
    / "isolation_forest_score_distribution.png"
)

# Save the completed figure
fig.savefig(
    score_distribution_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(
    "\nIsolation Forest score distribution saved to:",
    score_distribution_file,
)