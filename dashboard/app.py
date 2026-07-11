from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------

st.set_page_config(
    page_title="Salience Bot Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

WORKER_EVENTS_URL = os.getenv(
    "WORKER_EVENTS_URL",
    "https://salience-beacon-worker.mahin0710.workers.dev/events",
)

# Replace this default URL with your actual Cloudflare Pages website.

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🛡️ Salience Bot Detection Dashboard")

st.write(
    "Live dashboard for privacy-preserving telemetry events collected from the "
    "Cloudflare Pages website, processed by the Cloudflare Worker, saved in "
    "Cloudflare D1, and displayed for classification and risk analysis."
)

st.caption(
    "The dashboard shows rule-based risk scoring, supervised Gradient Boosting "
    "classification, unsupervised Isolation Forest anomaly detection, and "
    "offline model evaluation evidence."
)

st.caption(f"Worker events API: {WORKER_EVENTS_URL}")

st.markdown("---")


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def clean_live_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric fields before displaying live telemetry."""

    cleaned_df = df.copy()

    decimal_columns = [
        "request_interval_seconds",
        "error_rate",
        "isolation_decision_score",
    ]

    for column in decimal_columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce",
            ).round(4)

    if "risk_score" in cleaned_df.columns:
        cleaned_df["risk_score"] = (
            pd.to_numeric(
                cleaned_df["risk_score"],
                errors="coerce",
            )
            .clip(lower=0, upper=100)
            .round(0)
            .astype("Int64")
        )

    integer_columns = [
        "has_favicon_request",
        "requested_robots_txt",
        "pages_per_session",
        "cipher_suite_count",
        "extension_count",
        "sni_present",
        "anomaly_detected",
    ]

    for column in integer_columns:
        if column in cleaned_df.columns:
            cleaned_df[column] = pd.to_numeric(
                cleaned_df[column],
                errors="coerce",
            ).astype("Int64")

    return cleaned_df


def format_prediction_name(value: object) -> str:
    """Convert internal supervised labels into readable labels."""

    display_names = {
        "human": "Human",
        "good_bot": "Good Bot",
        "bad_bot": "Bad Bot",
        "scanner": "Scanner",
        "scanner_like": "Scanner",
        "bad_bot_or_scanner": "Bad Bot / Scanner",
        "azure_api_error": "API Error",
        "unknown": "Unknown",
    }

    normalized_value = str(value).strip().lower()

    return display_names.get(
        normalized_value,
        normalized_value.replace("_", " ").title(),
    )


def format_isolation_name(value: object) -> str:
    """Convert Isolation Forest labels into readable labels."""

    display_names = {
        "normal": "Normal",
        "anomaly": "Anomaly",
        "unknown": "Unknown",
        "none": "Unknown",
        "nan": "Unknown",
    }

    normalized_value = str(value).strip().lower()

    return display_names.get(
        normalized_value,
        normalized_value.replace("_", " ").title(),
    )


def format_risk_name(value: object) -> str:
    """Convert internal risk labels into readable labels."""

    display_names = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "unknown": "Unknown",
    }

    normalized_value = str(value).strip().lower()

    return display_names.get(
        normalized_value,
        normalized_value.replace("_", " ").title(),
    )


def anomaly_boolean_series(df: pd.DataFrame) -> pd.Series:
    """Convert stored anomaly values into true or false values."""

    if "anomaly_detected" not in df.columns:
        return pd.Series(False, index=df.index)

    return (
        df["anomaly_detected"]
        .fillna(0)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes"])
    )


# ---------------------------------------------------------
# LIVE WORKER EVENTS
# ---------------------------------------------------------

try:
    response = requests.get(
        WORKER_EVENTS_URL,
        timeout=10,
    )

    response.raise_for_status()

    payload = response.json()
    events = payload.get("events", [])

    if not events:
        st.warning(
            "No live telemetry events were found. "
            "Use the website test buttons to create events."
        )

    else:
        df = pd.DataFrame(events)
        df = clean_live_dataframe(df)

        # -------------------------------------------------
        # SUMMARY METRICS
        # -------------------------------------------------

        st.subheader("Live Prototype Summary")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Events",
                len(df),
            )

        with col2:
            if "risk_level" in df.columns:
                high_risk_events = int(
                    (
                        df["risk_level"]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        == "high"
                    ).sum()
                )
            else:
                high_risk_events = "N/A"

            st.metric(
                "High-Risk Events",
                high_risk_events,
            )

        with col3:
            if "risk_score" in df.columns:
                average_risk = pd.to_numeric(
                    df["risk_score"],
                    errors="coerce",
                ).mean()

                if pd.notna(average_risk):
                    average_risk_display = f"{average_risk:.2f} / 100"
                else:
                    average_risk_display = "N/A"
            else:
                average_risk_display = "N/A"

            st.metric(
                "Average Risk Score",
                average_risk_display,
            )

        with col4:
            anomaly_count = int(
                anomaly_boolean_series(df).sum()
            )

            st.metric(
                "Detected Anomalies",
                anomaly_count,
            )

        with col5:
            if "page_path" in df.columns:
                unique_paths = int(
                    df["page_path"].nunique()
                )
            else:
                unique_paths = "N/A"

            st.metric(
                "Unique Paths",
                unique_paths,
            )

        st.markdown("---")


        # -------------------------------------------------
        # DETECTION CHARTS
        # -------------------------------------------------

        st.subheader("Live Detection Results")

        chart_col1, chart_col2, chart_col3 = st.columns(3)


        # -------------------------------------------------
        # SUPERVISED GRADIENT BOOSTING CHART
        # -------------------------------------------------

        with chart_col1:
            if "worker_prediction" in df.columns:
                prediction_data = df[
                    df["worker_prediction"]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .isin(
                        [
                            "human",
                            "good_bot",
                            "bad_bot",
                            "scanner",
                            "scanner_like",
                            "bad_bot_or_scanner",
                        ]
                    )
                ].copy()

                if prediction_data.empty:
                    st.info(
                        "No valid supervised predictions are available."
                    )

                else:
                    prediction_counts = (
                        prediction_data["worker_prediction"]
                        .astype(str)
                        .str.lower()
                        .value_counts()
                        .rename_axis("worker_prediction")
                        .reset_index(name="count")
                    )

                    prediction_counts["Prediction"] = (
                        prediction_counts["worker_prediction"]
                        .apply(format_prediction_name)
                    )

                    prediction_chart = (
                        alt.Chart(prediction_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Prediction:N",
                                title="Prediction",
                                sort="-y",
                                axis=alt.Axis(
                                    labelAngle=0,
                                    labelFontSize=11,
                                    labelFontWeight="normal",
                                    labelLimit=120,
                                    titleFontSize=13,
                                    titleFontWeight="bold",
                                ),
                            ),
                            y=alt.Y(
                                "count:Q",
                                title="Number of Events",
                                axis=alt.Axis(
                                    labelFontSize=11,
                                    titleFontSize=13,
                                    titleFontWeight="bold",
                                    tickMinStep=1,
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "Prediction:N",
                                    title="Prediction",
                                ),
                                alt.Tooltip(
                                    "count:Q",
                                    title="Events",
                                ),
                            ],
                        )
                        .properties(
                            height=320,
                            title=alt.TitleParams(
                                text="Supervised Predictions",
                                subtitle="Gradient Boosting",
                                anchor="middle",
                                fontSize=18,
                                fontWeight="bold",
                                offset=15,
                            ),
                        )
                    )

                    st.altair_chart(
                        prediction_chart,
                        use_container_width=True,
                    )


        # -------------------------------------------------
        # UNSUPERVISED ISOLATION FOREST CHART
        # -------------------------------------------------

        with chart_col2:
            if "isolation_prediction" in df.columns:
                isolation_values = (
                    df["isolation_prediction"]
                    .fillna("unknown")
                    .astype(str)
                    .str.lower()
                )

                isolation_data = df[
                    isolation_values.isin(
                        [
                            "normal",
                            "anomaly",
                        ]
                    )
                ].copy()

                if isolation_data.empty:
                    st.info(
                        "No Isolation Forest results are available yet. "
                        "Generate a new event after deploying the updated Worker."
                    )

                else:
                    isolation_counts = (
                        isolation_data["isolation_prediction"]
                        .astype(str)
                        .str.lower()
                        .value_counts()
                        .rename_axis("isolation_prediction")
                        .reset_index(name="count")
                    )

                    isolation_counts["Isolation Result"] = (
                        isolation_counts["isolation_prediction"]
                        .apply(format_isolation_name)
                    )

                    isolation_chart = (
                        alt.Chart(isolation_counts)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Isolation Result:N",
                                title="Isolation Result",
                                sort=[
                                    "Normal",
                                    "Anomaly",
                                ],
                                axis=alt.Axis(
                                    labelAngle=0,
                                    labelFontSize=11,
                                    labelFontWeight="normal",
                                    titleFontSize=13,
                                    titleFontWeight="bold",
                                ),
                            ),
                            y=alt.Y(
                                "count:Q",
                                title="Number of Events",
                                axis=alt.Axis(
                                    labelFontSize=11,
                                    titleFontSize=13,
                                    titleFontWeight="bold",
                                    tickMinStep=1,
                                ),
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "Isolation Result:N",
                                    title="Result",
                                ),
                                alt.Tooltip(
                                    "count:Q",
                                    title="Events",
                                ),
                            ],
                        )
                        .properties(
                            height=320,
                            title=alt.TitleParams(
                                text="Unsupervised Results",
                                subtitle="Isolation Forest",
                                anchor="middle",
                                fontSize=18,
                                fontWeight="bold",
                                offset=15,
                            ),
                        )
                    )

                    st.altair_chart(
                        isolation_chart,
                        use_container_width=True,
                    )


        # -------------------------------------------------
        # RULE-BASED RISK CHART
        # -------------------------------------------------

        with chart_col3:
            if "risk_level" in df.columns:
                risk_counts = (
                    df["risk_level"]
                    .fillna("unknown")
                    .astype(str)
                    .str.lower()
                    .value_counts()
                    .rename_axis("risk_level")
                    .reset_index(name="count")
                )

                risk_counts["Risk Level"] = (
                    risk_counts["risk_level"]
                    .apply(format_risk_name)
                )

                risk_chart = (
                    alt.Chart(risk_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Risk Level:N",
                            title="Risk Level",
                            sort=[
                                "High",
                                "Medium",
                                "Low",
                                "Unknown",
                            ],
                            axis=alt.Axis(
                                labelAngle=0,
                                labelFontSize=11,
                                labelFontWeight="normal",
                                titleFontSize=13,
                                titleFontWeight="bold",
                            ),
                        ),
                        y=alt.Y(
                            "count:Q",
                            title="Number of Events",
                            axis=alt.Axis(
                                labelFontSize=11,
                                titleFontSize=13,
                                titleFontWeight="bold",
                                tickMinStep=1,
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "Risk Level:N",
                                title="Risk Level",
                            ),
                            alt.Tooltip(
                                "count:Q",
                                title="Events",
                            ),
                        ],
                    )
                    .properties(
                        height=320,
                        title=alt.TitleParams(
                            text="Rule-Based Risk Levels",
                            subtitle="Cloudflare Worker",
                            anchor="middle",
                            fontSize=18,
                            fontWeight="bold",
                            offset=15,
                        ),
                    )
                )

                st.altair_chart(
                    risk_chart,
                    use_container_width=True,
                )

        st.caption(
            "API errors are excluded from the supervised prediction chart "
            "so the chart shows only detection classifications."
        )

        st.markdown("---")


        # -------------------------------------------------
        # LATEST EVENTS TABLE
        # -------------------------------------------------

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

            # Supervised Gradient Boosting result.
            "worker_prediction",

            # Unsupervised Isolation Forest results.
            "isolation_prediction",
            "anomaly_detected",
            "isolation_decision_score",

            # Rule-based Worker results.
            "risk_score",
            "risk_level",
            "action",
        ]

        existing_columns = [
            column
            for column in preferred_columns
            if column in df.columns
        ]

        display_df = df[existing_columns].copy()

        if "worker_prediction" in display_df.columns:
            display_df["worker_prediction"] = (
                display_df["worker_prediction"]
                .apply(format_prediction_name)
            )

        if "isolation_prediction" in display_df.columns:
            display_df["isolation_prediction"] = (
                display_df["isolation_prediction"]
                .apply(format_isolation_name)
            )

        if "risk_level" in display_df.columns:
            display_df["risk_level"] = (
                display_df["risk_level"]
                .apply(format_risk_name)
            )

        display_df = display_df.rename(
            columns={
                "id": "ID",
                "timestamp": "Timestamp",
                "page_path": "Page Path",
                "interaction_type": "Interaction Type",
                "scroll_depth_category": "Scroll Depth",
                "request_interval_seconds": "Request Interval",
                "user_agent_category": "User-Agent Category",
                "has_favicon_request": "Favicon Request",
                "requested_robots_txt": "Robots.txt Requested",
                "pages_per_session": "Pages per Session",
                "error_rate": "Error Rate",
                "tls_version": "TLS Version",
                "cipher_suite_count": "Cipher Suite Count",
                "extension_count": "Extension Count",
                "alpn": "ALPN",
                "sni_present": "SNI Present",
                "worker_prediction": "Supervised Prediction",
                "isolation_prediction": "Isolation Result",
                "anomaly_detected": "Anomaly Detected",
                "isolation_decision_score": "Isolation Score",
                "risk_score": "Risk Score",
                "risk_level": "Risk Level",
                "action": "Action",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


        # -------------------------------------------------
        # RAW EVENT DATA
        # -------------------------------------------------

        with st.expander("Raw Event Data"):
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )


except requests.exceptions.RequestException as error:
    st.error(
        "Could not load live telemetry data from the Cloudflare Worker."
    )

    st.write(error)


except ValueError as error:
    st.error(
        "The Cloudflare Worker returned data that could not be read as JSON."
    )

    st.write(error)


# ---------------------------------------------------------
# MODEL EVALUATION RESULTS
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Model Evaluation Results")

st.write(
    "These metrics come from the offline test split of the synthetic dataset. "
    "The current accuracy, precision, recall, F1-score, and confusion matrix "
    "describe the supervised Gradient Boosting classifier. The live Isolation "
    "Forest normal-versus-anomaly results are shown in the dashboard above."
)


# ---------------------------------------------------------
# FIND MODEL EVALUATION FILE
# ---------------------------------------------------------

evaluation_path = (
    Path(__file__).parent
    / "dashboard"
    / "model_evaluation.json"
)

if not evaluation_path.exists():
    evaluation_path = (
        Path(__file__).parent
        / "model_evaluation.json"
    )


# ---------------------------------------------------------
# DISPLAY MODEL EVALUATION
# ---------------------------------------------------------

if evaluation_path.exists():
    with open(
        evaluation_path,
        "r",
        encoding="utf-8",
    ) as file:
        evaluation = json.load(file)

    st.markdown("### Overall Supervised Model Performance")

    (
        eval_col1,
        eval_col2,
        eval_col3,
        eval_col4,
        eval_col5,
    ) = st.columns(5)

    with eval_col1:
        st.metric(
            "Accuracy",
            f"{evaluation.get('accuracy', 0) * 100:.2f}%",
        )

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


    # -----------------------------------------------------
    # MODEL DETAILS
    # -----------------------------------------------------

    st.markdown("### Model Details")

    detail_col1, detail_col2, detail_col3, detail_col4 = (
        st.columns(4)
    )

    with detail_col1:
        st.info(
            f"Dataset: {evaluation.get('dataset', 'N/A')}"
        )

    with detail_col2:
        st.info(
            f"Test Records: {evaluation.get('test_records', 'N/A')}"
        )

    with detail_col3:
        st.info(
            f"Feature Count: {evaluation.get('feature_count', 'N/A')}"
        )

    with detail_col4:
        st.info(
            f"scikit-learn: "
            f"{evaluation.get('scikit_learn_version', 'N/A')}"
        )

    st.info(
        f"Model Family: "
        f"{evaluation.get('model_family', 'N/A')}"
    )


    # -----------------------------------------------------
    # CLASS-LEVEL METRICS
    # -----------------------------------------------------

    st.markdown("### Class-Level Metrics")

    class_metrics = evaluation.get(
        "class_metrics",
        [],
    )

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

        if "Class" in class_df.columns:
            class_df["Class"] = (
                class_df["Class"]
                .apply(format_prediction_name)
            )

        for metric_column in [
            "Precision",
            "Recall",
            "F1-Score",
        ]:
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
        st.info(
            "Class-level metrics are not available."
        )


    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    st.markdown("### Confusion Matrix")

    confusion_matrix = evaluation.get(
        "confusion_matrix",
        [],
    )

    labels = evaluation.get(
        "confusion_matrix_labels",
        [],
    )

    if confusion_matrix and labels:
        readable_labels = [
            format_prediction_name(label)
            for label in labels
        ]

        confusion_matrix_df = pd.DataFrame(
            confusion_matrix,
            index=[
                f"Actual: {label}"
                for label in readable_labels
            ],
            columns=[
                f"Predicted: {label}"
                for label in readable_labels
            ],
        )

        st.dataframe(
            confusion_matrix_df,
            use_container_width=True,
        )

    else:
        st.info(
            "Confusion matrix data is not available."
        )

else:
    st.warning(
        "The model evaluation file was not found. "
        "Make sure model_evaluation.json is saved in "
        "dashboard/dashboard/model_evaluation.json or "
        "dashboard/model_evaluation.json."
    )


# ---------------------------------------------------------
# PRIVACY NOTE
# ---------------------------------------------------------

st.markdown("---")
st.subheader("Privacy-Preserving Telemetry Note")

st.write(
    "The system stores minimized telemetry summaries only. "
    "It does not store raw IP addresses, names, emails, passwords, "
    "cookies, authentication tokens, private user content, exact "
    "location, or invasive browser fingerprinting."
)