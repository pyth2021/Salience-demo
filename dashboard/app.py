from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="Salience Bot Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# -----------------------------
# Header
# -----------------------------
st.title("🛡️ Salience Bot Detection Dashboard")

st.write(
    "Live dashboard for privacy-preserving telemetry events collected from the "
    "Cloudflare Pages website, processed by the Cloudflare Worker, saved in "
    "Cloudflare D1, and displayed for classification and risk analysis."
)

st.caption(
    "This dashboard focuses on live telemetry, prediction results, risk scoring, "
    "and model evaluation evidence."
)

st.markdown("---")


# -----------------------------
# Worker API Configuration
# -----------------------------
WORKER_EVENTS_URL = os.getenv(
    "WORKER_EVENTS_URL",
    "https://salience-beacon-worker.mahin0710.workers.dev/events",
)

st.caption(f"Worker events API: {WORKER_EVENTS_URL}")


# -----------------------------
# Helper Functions
# -----------------------------
def clean_live_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns for a cleaner dashboard table."""

    cleaned_df = df.copy()

    numeric_columns_2dp = [
        "request_interval_seconds",
        "error_rate",
    ]

    for column in numeric_columns_2dp:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce",
            ).round(2)

    if "risk_score" in cleaned_df.columns:
        cleaned_df["risk_score"] = pd.to_numeric(
            cleaned_df["risk_score"],
            errors="coerce",
        ).round(0).astype("Int64")

    integer_columns = [
        "has_favicon_request",
        "requested_robots_txt",
        "pages_per_session",
        "cipher_suite_count",
        "extension_count",
        "sni_present",
    ]

    for column in integer_columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce",
            ).astype("Int64")

    return cleaned_df


# -----------------------------
# Live Worker Events
# -----------------------------
try:
    response = requests.get(WORKER_EVENTS_URL, timeout=10)
    response.raise_for_status()

    payload = response.json()
    events = payload.get("events", [])

    if not events:
        st.warning("No live telemetry events found yet. Use the website test buttons first.")

    else:
        df = pd.DataFrame(events)
        df = clean_live_dataframe(df)

        # -----------------------------
        # Summary Metrics
        # -----------------------------
        st.subheader("Live Prototype Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Events", len(df))

        with col2:
            if "risk_level" in df.columns:
                high_risk = df[df["risk_level"].str.lower() == "high"].shape[0]
            else:
                high_risk = "N/A"
            st.metric("High-Risk Events", high_risk)

        with col3:
            if "risk_score" in df.columns:
                avg_risk = pd.to_numeric(
                    df["risk_score"],
                    errors="coerce",
                ).mean()

                if pd.notna(avg_risk):
                    avg_risk_display = f"{avg_risk:.2f}"
                else:
                    avg_risk_display = "N/A"
            else:
                avg_risk_display = "N/A"

            st.metric("Average Risk Score", avg_risk_display)

        with col4:
            unique_paths = df["page_path"].nunique() if "page_path" in df.columns else "N/A"
            st.metric("Unique Paths", unique_paths)

        st.markdown("---")

        # -----------------------------
        # Charts
        # -----------------------------
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if "worker_prediction" in df.columns:
                st.subheader("Worker Prediction Counts")
                st.bar_chart(df["worker_prediction"].value_counts())

        with chart_col2:
            if "risk_level" in df.columns:
                st.subheader("Risk Level Counts")
                st.bar_chart(df["risk_level"].value_counts())

        st.markdown("---")

        # -----------------------------
        # Latest Events Table
        # -----------------------------
        st.subheader("Latest Live Telemetry Events")

        preferred_columns = [
            "id",
            "timestamp",
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
            "worker_prediction",
            "risk_score",
            "risk_level",
            "action",
        ]

        existing_columns = [
            column for column in preferred_columns if column in df.columns
        ]

        st.dataframe(
            df[existing_columns],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Raw Event Data"):
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

except requests.exceptions.RequestException as error:
    st.error("Could not load live telemetry data from the Cloudflare Worker.")
    st.write(error)


# -----------------------------
# Model Evaluation Results
# -----------------------------
st.markdown("---")
st.subheader("Model Evaluation Results")

st.write(
    "These metrics come from the offline test split of the synthetic dataset. "
    "Live events are used for prediction, while evaluation metrics show how the "
    "trained model performed during testing."
)

evaluation_path = Path(__file__).parent / "dashboard" / "model_evaluation.json"

if not evaluation_path.exists():
    evaluation_path = Path(__file__).parent / "model_evaluation.json"

if evaluation_path.exists():
    with open(evaluation_path, "r", encoding="utf-8") as file:
        evaluation = json.load(file)

    st.markdown("### Overall Performance")

    eval_col1, eval_col2, eval_col3, eval_col4, eval_col5 = st.columns(5)

    with eval_col1:
        st.metric("Accuracy", f"{evaluation.get('accuracy', 0) * 100:.2f}%")

    with eval_col2:
        st.metric(
            "Weighted Precision",
            f"{evaluation.get('weighted_precision', 0) * 100:.2f}%",
        )

    with eval_col3:
        st.metric(
            "Weighted Recall",
            f"{evaluation.get('weighted_recall', 0) * 100:.2f}%",
        )

    with eval_col4:
        st.metric(
            "Weighted F1-Score",
            f"{evaluation.get('weighted_f1', 0) * 100:.2f}%",
        )

    with eval_col5:
        st.metric(
            "False Positive Rate",
            f"{evaluation.get('false_positive_rate', 0) * 100:.2f}%",
        )

    st.markdown("### Model Details")

    detail_col1, detail_col2, detail_col3 = st.columns(3)

    with detail_col1:
        st.info(f"Dataset: {evaluation.get('dataset', 'N/A')}")

    with detail_col2:
        st.info(f"Test Records: {evaluation.get('test_records', 'N/A')}")

    with detail_col3:
        st.info(f"Model Family: {evaluation.get('model_family', 'N/A')}")

    st.markdown("### Class-Level Metrics")

    class_metrics = evaluation.get("class_metrics", [])

    if class_metrics:
        class_df = pd.DataFrame(class_metrics)

        class_df = class_df.rename(
            columns={
                "class": "Class",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1-Score",
            }
        )

        for metric_column in ["Precision", "Recall", "F1-Score"]:
            if metric_column in class_df.columns:
                class_df[metric_column] = pd.to_numeric(
                    class_df[metric_column],
                    errors="coerce",
                ).round(4)

        st.dataframe(
            class_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("Class-level metrics are not available.")

    st.markdown("### Confusion Matrix")

    confusion_matrix = evaluation.get("confusion_matrix", [])
    labels = evaluation.get("confusion_matrix_labels", [])

    if confusion_matrix and labels:
        cm_df = pd.DataFrame(
            confusion_matrix,
            index=[f"Actual: {label}" for label in labels],
            columns=[f"Predicted: {label}" for label in labels],
        )

        st.dataframe(
            cm_df,
            use_container_width=True,
        )

    else:
        st.info("Confusion matrix is not available.")

else:
    st.warning(
        "Model evaluation file was not found. Make sure model_evaluation.json is saved in "
        "dashboard/dashboard/model_evaluation.json or dashboard/model_evaluation.json."
    )


# -----------------------------
# Privacy Note
# -----------------------------
st.markdown("---")
st.subheader("Privacy-Preserving Telemetry Note")

st.write(
    "The system stores minimized telemetry summaries only. It does not store raw IP addresses, "
    "names, emails, passwords, cookies, authentication tokens, private user content, exact "
    "location, or invasive browser fingerprinting."
)