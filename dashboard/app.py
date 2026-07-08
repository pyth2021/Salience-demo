from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Salience Bot Detection Dashboard", layout="wide")

st.title("Salience Bot Detection Dashboard")

st.write(
    "Live dashboard for events collected from the Cloudflare Pages website, "
    "processed by the Cloudflare Worker, saved in D1, and displayed here."
)

# -----------------------------
# Project Links
# -----------------------------
st.markdown("### Live Project Links")

link_col1, link_col2, link_col3, link_col4 = st.columns(4)

with link_col1:
    st.markdown("[Live Website](https://salience-demo.pages.dev)")

with link_col2:
    st.markdown("[Worker Events API](https://salience-beacon-worker.mahin0710.workers.dev/events)")

with link_col3:
    st.markdown("[Streamlit Dashboard](https://salience-demo-we5wxbfsagmuuzj3vyjvog.streamlit.app/)")

with link_col4:
    st.markdown("[GitHub Repository](https://github.com/pyth2021/Salience-demo)")

st.markdown("---")

# -----------------------------
# Team and Project Context
# -----------------------------
st.subheader("Team and Project Context")

team_col, ctx_col = st.columns(2)

with team_col:
    st.markdown("### Capstone Team")
    st.markdown("- **Team Member:** Mahin Chowdhury")
    st.markdown("- **Team Member:** Naveen Saranya Patro Behara")
    st.markdown("- **Team Member:** Olamide Akinyosola")
    st.markdown("- **Team Member:** Suma Madasu")

with ctx_col:
    st.markdown("### Project Context")
    st.markdown("- **Institution:** Humber Polytechnic, Ontario, Canada")
    st.markdown("- **Faculty:** Faculty of Applied Sciences & Technology")
    st.markdown("- **Program:** Cybersecurity and Artificial Intelligence")
    st.markdown("- **Project Sponsor:** Salience Enterprises Inc.")
    st.markdown("- **Industry Supervisor:** Abdullah Ali Syed")
    st.markdown("- **Course Instructor:** Asama Nseaf")
    st.markdown("- **Project Phase:** Live Prototype Implementation")

st.markdown("---")

# -----------------------------
# Live Worker Events
# -----------------------------
WORKER_EVENTS_URL = os.getenv(
    "WORKER_EVENTS_URL",
    "https://salience-beacon-worker.mahin0710.workers.dev/events",
)

st.caption(f"Worker events API: {WORKER_EVENTS_URL}")

try:
    response = requests.get(WORKER_EVENTS_URL, timeout=10)
    response.raise_for_status()

    payload = response.json()
    events = payload.get("events", [])

    if not events:
        st.warning("No live telemetry events found yet. Use the website test buttons first.")
    else:
        df = pd.DataFrame(events)

        st.subheader("Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total events", len(df))

        with col2:
            if "risk_level" in df.columns:
                high_risk = df[df["risk_level"] == "high"].shape[0]
            else:
                high_risk = "N/A"
            st.metric("High-risk events", high_risk)

        with col3:
            if "risk_score" in df.columns:
                avg_risk = round(
                    pd.to_numeric(df["risk_score"], errors="coerce").mean(),
                    2,
                )
            else:
                avg_risk = "N/A"
            st.metric("Average risk score", avg_risk)

        with col4:
            unique_paths = df["page_path"].nunique() if "page_path" in df.columns else "N/A"
            st.metric("Unique paths", unique_paths)

        if "worker_prediction" in df.columns:
            st.subheader("Worker Prediction Counts")
            st.bar_chart(df["worker_prediction"].value_counts())

        if "risk_level" in df.columns:
            st.subheader("Risk Level Counts")
            st.bar_chart(df["risk_level"].value_counts())

        st.subheader("Latest Live Telemetry Events")

        preferred_columns = [
            "id",
            "created_at",
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

        existing_columns = [column for column in preferred_columns if column in df.columns]
        st.dataframe(df[existing_columns], use_container_width=True)

        with st.expander("Raw Event Data"):
            st.dataframe(df, use_container_width=True)

except requests.exceptions.RequestException as error:
    st.error("Could not load live telemetry data from the Cloudflare Worker.")
    st.write(error)

# -----------------------------
# Model Evaluation Results
# -----------------------------
st.markdown("---")
st.subheader("Model Evaluation Results")

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
    st.write(f"**Dataset:** {evaluation.get('dataset', 'N/A')}")
    st.write(f"**Test Records:** {evaluation.get('test_records', 'N/A')}")
    st.write(f"**Model Family:** {evaluation.get('model_family', 'N/A')}")

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

        st.dataframe(class_df, use_container_width=True)
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

        st.dataframe(cm_df, use_container_width=True)
    else:
        st.info("Confusion matrix is not available.")

else:
    st.warning(
        "Model evaluation file was not found. Make sure model_evaluation.json is saved in "
        "dashboard/dashboard/model_evaluation.json or dashboard/model_evaluation.json."
    )