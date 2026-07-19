from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st


# -----------------------------------------------------------------------------
# PAGE AND CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Salience Bot Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

WORKER_EVENTS_URL = os.getenv(
    "WORKER_EVENTS_URL",
    "https://salience-beacon-worker.mahin0710.workers.dev/events",
)
REQUEST_TIMEOUT_SECONDS = 10


# -----------------------------------------------------------------------------
# GENERAL HELPERS
# -----------------------------------------------------------------------------

READABLE_NAMES = {
    "human": "Human",
    "good_bot": "Good Bot",
    "bad_bot": "Bad Bot",
    "scanner": "Scanner",
    "normal": "Normal",
    "anomaly": "Anomaly",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "azure_api_error": "API Error",
    "unknown": "Unknown",
    "none": "Unknown",
    "nan": "Unknown",
}


def readable_name(value: object) -> str:
    """Convert internal values into readable display text."""
    normalized = str(value).strip().lower()
    return READABLE_NAMES.get(normalized, normalized.replace("_", " ").title())


def as_percentage(value: Any) -> str:
    """Format a decimal value as a percentage."""
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def as_number(value: Any, decimals: int = 4) -> str:
    """Format a numeric value safely."""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def anomaly_series(dataframe: pd.DataFrame) -> pd.Series:
    """Convert stored anomaly values into Boolean values."""
    if "anomaly_detected" not in dataframe.columns:
        return pd.Series(False, index=dataframe.index, dtype=bool)

    return (
        dataframe["anomaly_detected"]
        .fillna(0)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes"])
    )


def clean_live_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean live database events before displaying them."""
    cleaned = dataframe.copy()

    # Older records may contain page_path but not page_category.
    if "page_category" not in cleaned.columns and "page_path" in cleaned.columns:
        cleaned["page_category"] = cleaned["page_path"]

    decimal_columns = [
        "request_interval_seconds",
        "error_rate",
        "isolation_decision_score",
    ]
    integer_columns = [
        "has_favicon_request",
        "requested_robots_txt",
        "pages_per_session",
        "cipher_suite_count",
        "extension_count",
        "sni_present",
        "anomaly_detected",
    ]

    for column in decimal_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column], errors="coerce"
            ).round(6)

    for column in integer_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column], errors="coerce"
            ).astype("Int64")

    if "risk_score" in cleaned.columns:
        cleaned["risk_score"] = (
            pd.to_numeric(cleaned["risk_score"], errors="coerce")
            .clip(lower=0, upper=100)
            .round()
            .astype("Int64")
        )

    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(
            cleaned["timestamp"], errors="coerce", utc=True
        )

    return cleaned


def count_dataframe(
    dataframe: pd.DataFrame,
    source_column: str,
    display_column: str,
    allowed_values: list[str] | None = None,
) -> pd.DataFrame:
    """Build a count table for one categorical column."""
    if source_column not in dataframe.columns:
        return pd.DataFrame()

    normalized = (
        dataframe[source_column].fillna("unknown").astype(str).str.lower()
    )

    if allowed_values:
        normalized = normalized[normalized.isin(allowed_values)]

    if normalized.empty:
        return pd.DataFrame()

    counts = (
        normalized.value_counts()
        .rename_axis("internal_value")
        .reset_index(name="count")
    )
    counts[display_column] = counts["internal_value"].apply(readable_name)
    return counts


def count_chart(
    dataframe: pd.DataFrame,
    display_column: str,
    title: str,
    subtitle: str,
    sort_order: list[str] | str,
) -> alt.Chart:
    """Create a standard event-count bar chart."""
    return (
        alt.Chart(dataframe)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{display_column}:N",
                title=display_column,
                sort=sort_order,
                axis=alt.Axis(labelAngle=0, labelLimit=130),
            ),
            y=alt.Y(
                "count:Q",
                title="Number of Events",
                axis=alt.Axis(tickMinStep=1),
            ),
            tooltip=[
                alt.Tooltip(f"{display_column}:N", title=display_column),
                alt.Tooltip("count:Q", title="Events"),
            ],
        )
        .properties(
            height=320,
            title=alt.TitleParams(
                text=title,
                subtitle=subtitle,
                anchor="middle",
                fontSize=18,
                fontWeight="bold",
                offset=15,
            ),
        )
    )


def display_count_chart(
    dataframe: pd.DataFrame,
    source_column: str,
    display_column: str,
    title: str,
    subtitle: str,
    sort_order: list[str] | str,
    allowed_values: list[str] | None = None,
) -> None:
    """Create and display a chart or an unavailable-data message."""
    counts = count_dataframe(
        dataframe,
        source_column,
        display_column,
        allowed_values,
    )

    if counts.empty:
        st.info(f"No {display_column.lower()} data is available.")
        return

    chart = count_chart(
        counts,
        display_column,
        title,
        subtitle,
        sort_order,
    )
    st.altair_chart(chart, use_container_width=True)


def show_metrics(metrics: dict[str, Any], definitions: list[tuple[str, str]]) -> None:
    """Display percentage metrics in equally sized columns."""
    columns = st.columns(len(definitions))

    for column, (label, key) in zip(columns, definitions):
        with column:
            st.metric(label, as_percentage(metrics.get(key)))


def show_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> None:
    """Display a confusion matrix as a readable table."""
    if not matrix or not labels:
        st.info("Confusion matrix data is not available.")
        return

    readable_labels = [readable_name(label) for label in labels]
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[f"Actual: {label}" for label in readable_labels],
        columns=[f"Predicted: {label}" for label in readable_labels],
    )
    st.dataframe(matrix_dataframe, use_container_width=True)


def find_evaluation_file() -> Path | None:
    """Find model_evaluation.json in a supported dashboard location."""
    current_directory = Path(__file__).resolve().parent
    candidates = [
        current_directory / "dashboard" / "model_evaluation.json",
        current_directory / "model_evaluation.json",
    ]

    return next((candidate for candidate in candidates if candidate.exists()), None)


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------

st.title("🛡️ Salience Bot Detection Dashboard")

st.write(
    "Live dashboard for privacy-preserving telemetry events collected from "
    "the Cloudflare Pages website, processed by the Cloudflare Worker, "
    "classified by the Azure ML API, and stored in Cloudflare D1."
)

st.caption(
    "The live section shows Gradient Boosting classifications, Isolation "
    "Forest anomaly results, and final Worker risk decisions. The offline "
    "section shows results from the independent synthetic holdout dataset."
)

st.caption(f"Worker events API: {WORKER_EVENTS_URL}")
st.markdown("---")


# -----------------------------------------------------------------------------
# LIVE WORKER EVENTS
# -----------------------------------------------------------------------------

try:
    response = requests.get(
        WORKER_EVENTS_URL,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    events = payload.get("events", [])

    if not isinstance(events, list):
        raise ValueError("The events field must contain a list.")

    if not events:
        st.warning(
            "No live telemetry events were found. Use the website test "
            "buttons to create events."
        )
    else:
        live_dataframe = clean_live_dataframe(pd.DataFrame(events))

        # ---------------------------------------------------------------------
        # LIVE SUMMARY
        # ---------------------------------------------------------------------

        st.subheader("Live Prototype Summary")

        high_risk_events: int | str = "N/A"
        if "risk_level" in live_dataframe.columns:
            high_risk_events = int(
                live_dataframe["risk_level"]
                .fillna("")
                .astype(str)
                .str.lower()
                .eq("high")
                .sum()
            )

        average_risk_display = "N/A"
        if "risk_score" in live_dataframe.columns:
            average_risk = pd.to_numeric(
                live_dataframe["risk_score"], errors="coerce"
            ).mean()

            if pd.notna(average_risk):
                average_risk_display = f"{average_risk:.2f} / 100"

        unique_categories: int | str = "N/A"
        if "page_category" in live_dataframe.columns:
            unique_categories = int(
                live_dataframe["page_category"].dropna().nunique()
            )

        summary_values = [
            ("Recent Events Loaded", len(live_dataframe)),
            ("High-Risk Events", high_risk_events),
            ("Average Risk Score", average_risk_display),
            ("Detected Anomalies", int(anomaly_series(live_dataframe).sum())),
            ("Unique Page Categories", unique_categories),
        ]

        for column, (label, value) in zip(st.columns(5), summary_values):
            with column:
                st.metric(label, value)

        st.caption(
            "The Worker returns the latest 100 stored events, so these "
            "metrics describe the recently loaded window rather than all "
            "database history."
        )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # LIVE CHARTS
        # ---------------------------------------------------------------------

        st.subheader("Live Detection Results")
        chart_columns = st.columns(3)

        with chart_columns[0]:
            display_count_chart(
                live_dataframe,
                source_column="worker_prediction",
                display_column="Prediction",
                title="Supervised Predictions",
                subtitle="Gradient Boosting",
                sort_order="-y",
                allowed_values=["human", "good_bot", "bad_bot", "scanner"],
            )

        with chart_columns[1]:
            display_count_chart(
                live_dataframe,
                source_column="isolation_prediction",
                display_column="Isolation Result",
                title="Unsupervised Results",
                subtitle="Isolation Forest",
                sort_order=["Normal", "Anomaly"],
                allowed_values=["normal", "anomaly"],
            )

        with chart_columns[2]:
            display_count_chart(
                live_dataframe,
                source_column="risk_level",
                display_column="Risk Level",
                title="Final Risk Levels",
                subtitle="Cloudflare Worker",
                sort_order=["High", "Medium", "Low", "Unknown"],
            )

        st.caption(
            "API errors and unknown values are excluded from the supervised "
            "chart so it shows only the four supported traffic classes."
        )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # LATEST EVENTS TABLE
        # ---------------------------------------------------------------------

        st.subheader("Latest Live Telemetry Events")

        preferred_columns = [
            "id",
            "timestamp",
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
            "worker_prediction",
            "isolation_prediction",
            "anomaly_detected",
            "isolation_decision_score",
            "risk_score",
            "risk_level",
            "action",
        ]

        column_names = {
            "id": "ID",
            "timestamp": "Timestamp",
            "page_category": "Page Category",
            "interaction_type": "Interaction Type",
            "scroll_depth_category": "Scroll Depth",
            "request_interval_seconds": "Request Interval (Seconds)",
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

        existing_columns = [
            column
            for column in preferred_columns
            if column in live_dataframe.columns
        ]
        display_dataframe = live_dataframe[existing_columns].copy()

        for column in [
            "worker_prediction",
            "isolation_prediction",
            "risk_level",
            "action",
        ]:
            if column in display_dataframe.columns:
                display_dataframe[column] = display_dataframe[column].apply(
                    readable_name
                )

        display_dataframe = display_dataframe.rename(columns=column_names)

        st.dataframe(
            display_dataframe,
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Raw Event Data"):
            st.dataframe(
                live_dataframe,
                use_container_width=True,
                hide_index=True,
            )

except requests.exceptions.RequestException as error:
    st.error(
        "Could not load live telemetry data from the Cloudflare Worker."
    )
    st.write(error)

except (ValueError, TypeError, json.JSONDecodeError) as error:
    st.error(
        "The Cloudflare Worker returned data that could not be processed."
    )
    st.write(error)


# -----------------------------------------------------------------------------
# OFFLINE MODEL EVALUATION
# -----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Offline Model Evaluation")

st.write(
    "Gradient Boosting and Isolation Forest were evaluated separately on "
    "the independent synthetic holdout dataset. Gradient Boosting predicts "
    "one of four traffic classes. Isolation Forest predicts whether an event "
    "is normal or anomalous compared with benign training traffic."
)

evaluation_path = find_evaluation_file()

if evaluation_path is None:
    st.warning(
        "The model evaluation file was not found. Expected "
        "model_evaluation.json inside dashboard/dashboard or dashboard."
    )
else:
    try:
        with evaluation_path.open("r", encoding="utf-8") as file:
            evaluation = json.load(file)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error("The model evaluation file could not be loaded.")
        st.write(error)
    else:
        gradient_section = evaluation.get("gradient_boosting", {})
        isolation_section = evaluation.get("isolation_forest", {})

        gradient_validation = gradient_section.get(
            "internal_validation", {}
        )
        gradient_final = gradient_section.get("final_holdout", {})
        isolation_final = isolation_section.get("final_holdout", {})

        # Support the older supervised-only evaluation format.
        if not gradient_final:
            gradient_final = {
                key: evaluation.get(key)
                for key in [
                    "accuracy",
                    "weighted_precision",
                    "weighted_recall",
                    "weighted_f1",
                    "false_positive_rate",
                    "class_metrics",
                    "confusion_matrix",
                    "confusion_matrix_labels",
                ]
            }

        gradient_tab, isolation_tab, details_tab = st.tabs(
            [
                "Gradient Boosting",
                "Isolation Forest",
                "Dataset and Model Details",
            ]
        )

        # ---------------------------------------------------------------------
        # GRADIENT BOOSTING TAB
        # ---------------------------------------------------------------------

        with gradient_tab:
            st.markdown("### Final Holdout Performance")

            show_metrics(
                gradient_final,
                [
                    ("Accuracy", "accuracy"),
                    ("Balanced Accuracy", "balanced_accuracy"),
                    ("Macro F1", "macro_f1"),
                    ("Weighted F1", "weighted_f1"),
                    ("False Positive Rate", "false_positive_rate"),
                ],
            )

            st.caption(
                "The final holdout has a realistic imbalanced class "
                "distribution and was not used to fit either model."
            )

            if gradient_validation:
                st.markdown("### Internal Validation Performance")

                show_metrics(
                    gradient_validation,
                    [
                        ("Accuracy", "accuracy"),
                        ("Balanced Accuracy", "balanced_accuracy"),
                        ("Macro F1", "macro_f1"),
                        ("Weighted F1", "weighted_f1"),
                    ],
                )

            st.markdown("### Class-Level Metrics")
            class_metrics = gradient_final.get("class_metrics", [])

            if not class_metrics:
                st.info("Class-level metrics are not available.")
            else:
                class_dataframe = pd.DataFrame(class_metrics).rename(
                    columns={
                        "class": "Class",
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1_score": "F1-Score",
                        "support": "Support",
                    }
                )

                if "Class" in class_dataframe.columns:
                    class_dataframe["Class"] = class_dataframe["Class"].apply(
                        readable_name
                    )

                st.dataframe(
                    class_dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Gradient Boosting Confusion Matrix")
            show_confusion_matrix(
                gradient_final.get("confusion_matrix", []),
                gradient_final.get("confusion_matrix_labels", []),
            )

        # ---------------------------------------------------------------------
        # ISOLATION FOREST TAB
        # ---------------------------------------------------------------------

        with isolation_tab:
            if not isolation_final:
                st.info(
                    "Isolation Forest offline metrics are not available in "
                    "this evaluation file."
                )
            else:
                st.markdown("### Final Holdout Anomaly Performance")

                show_metrics(
                    isolation_final,
                    [
                        ("Anomaly Precision", "anomaly_precision"),
                        ("Anomaly Recall", "anomaly_recall"),
                        ("Anomaly F1", "anomaly_f1"),
                        ("False Positive Rate", "false_positive_rate"),
                        ("False Negative Rate", "false_negative_rate"),
                    ],
                )

                anomaly_columns = st.columns(3)

                with anomaly_columns[0]:
                    st.metric(
                        "Predicted Anomaly Rate",
                        as_percentage(
                            isolation_final.get("predicted_anomaly_rate")
                        ),
                    )

                with anomaly_columns[1]:
                    st.metric(
                        "Ground-Truth Anomaly Rate",
                        as_percentage(
                            isolation_final.get("ground_truth_anomaly_rate")
                        ),
                    )

                # The threshold is stored in the main Isolation Forest section.
                threshold = isolation_section.get(
                    "anomaly_threshold",
                    isolation_final.get("anomaly_threshold"),
                )

                with anomaly_columns[2]:
                    st.metric(
                        "Anomaly Threshold",
                        as_number(threshold, decimals=6),
                    )

                st.caption(
                    "Lower Isolation Forest scores are more unusual. An "
                    "event is marked as an anomaly when its score is below "
                    "the calibrated threshold."
                )

                st.markdown("### Isolation Forest Confusion Matrix")
                show_confusion_matrix(
                    isolation_final.get("confusion_matrix", []),
                    isolation_final.get("confusion_matrix_labels", []),
                )

                st.markdown("### Anomaly Rates by Traffic Class")
                per_class_rates = isolation_final.get(
                    "per_class_anomaly_rates", []
                )

                if not per_class_rates:
                    st.info("Per-class anomaly rates are not available.")
                else:
                    rates_dataframe = pd.DataFrame(per_class_rates).rename(
                        columns={
                            "class": "Class",
                            "records": "Records",
                            "predicted_anomaly_rate": (
                                "Predicted Anomaly Rate (%)"
                            ),
                            "ground_truth_anomaly_rate": (
                                "Ground-Truth Anomaly Rate (%)"
                            ),
                        }
                    )

                    if "Class" in rates_dataframe.columns:
                        rates_dataframe["Class"] = rates_dataframe[
                            "Class"
                        ].apply(readable_name)

                    rate_columns = [
                        "Predicted Anomaly Rate (%)",
                        "Ground-Truth Anomaly Rate (%)",
                    ]

                    for column in rate_columns:
                        if column in rates_dataframe.columns:
                            rates_dataframe[column] = (
                                pd.to_numeric(
                                    rates_dataframe[column],
                                    errors="coerce",
                                )
                                * 100
                            ).round(2)

                    st.dataframe(
                        rates_dataframe,
                        use_container_width=True,
                        hide_index=True,
                    )

        # ---------------------------------------------------------------------
        # DATASET AND MODEL DETAILS TAB
        # ---------------------------------------------------------------------

        with details_tab:
            st.markdown("### Dataset Details")

            details = [
                (
                    "Development Records",
                    evaluation.get("development_records", "N/A"),
                ),
                (
                    "Development Training Records",
                    evaluation.get("development_training_records", "N/A"),
                ),
                (
                    "Internal Validation Records",
                    evaluation.get("internal_validation_records", "N/A"),
                ),
                (
                    "Final Holdout Records",
                    evaluation.get(
                        "final_holdout_records",
                        evaluation.get("test_records", "N/A"),
                    ),
                ),
            ]

            for column, (label, value) in zip(st.columns(4), details):
                with column:
                    st.metric(label, value)

            model_details = [
                ("Feature Count", evaluation.get("feature_count", "N/A")),
                (
                    "scikit-learn Version",
                    evaluation.get("scikit_learn_version", "N/A"),
                ),
                ("Random State", evaluation.get("random_state", "N/A")),
            ]

            for column, (label, value) in zip(
                st.columns(3), model_details
            ):
                with column:
                    st.metric(label, value)

            st.markdown("### Dataset Class Counts")

            development_counts = evaluation.get(
                "development_class_counts", {}
            )
            holdout_counts = evaluation.get(
                "final_holdout_class_counts", {}
            )
            class_names = sorted(
                set(development_counts) | set(holdout_counts)
            )

            if not class_names:
                st.info("Dataset class counts are not available.")
            else:
                count_rows = [
                    {
                        "Class": readable_name(class_name),
                        "Development Records": development_counts.get(
                            class_name, 0
                        ),
                        "Final Holdout Records": holdout_counts.get(
                            class_name, 0
                        ),
                    }
                    for class_name in class_names
                ]

                st.dataframe(
                    pd.DataFrame(count_rows),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Model Features")
            features = evaluation.get("features", [])

            if not features:
                st.info("The feature list is not available.")
            else:
                feature_dataframe = pd.DataFrame(
                    {
                        "Feature Number": range(1, len(features) + 1),
                        "Feature": features,
                    }
                )

                st.dataframe(
                    feature_dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("### Isolation Forest Training Design")

            st.write(
                "Isolation Forest was fitted without target labels. Human "
                "and Good Bot labels were used only to select benign "
                "development records before fitting."
            )

            isolation_training_records = isolation_section.get(
                "training_records", "N/A"
            )
            threshold_records = isolation_section.get(
                "threshold_calibration_records", "N/A"
            )
            normal_labels = isolation_section.get(
                "training_labels_used_for_selection",
                ["human", "good_bot"],
            )

            st.info(
                f"Benign training records: {isolation_training_records}\n\n"
                f"Threshold-calibration records: {threshold_records}\n\n"
                f"Normal training classes: "
                f"{', '.join(readable_name(label) for label in normal_labels)}"
            )


# -----------------------------------------------------------------------------
# PRIVACY NOTE
# -----------------------------------------------------------------------------

st.markdown("---")
st.subheader("Privacy-Preserving Telemetry Note")

st.write(
    "The system stores minimized telemetry summaries only. It does not "
    "store raw IP addresses, names, emails, passwords, cookies, "
    "authentication tokens, private user content, exact location, full "
    "URLs, or invasive browser fingerprints."
)