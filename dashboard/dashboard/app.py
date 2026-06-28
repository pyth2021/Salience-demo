import pandas as pd
import streamlit as st
import requests

st.set_page_config(
    page_title="Salience Bot Detection Dashboard",
    layout="wide"
)

st.title("Salience Bot Detection Dashboard")

st.write(
    "This dashboard shows live telemetry events collected from the Cloudflare Pages website, "
    "processed by the Cloudflare Worker, saved in Cloudflare D1, and displayed here."
)

WORKER_EVENTS_URL = "https://salience-beacon-worker.mahin0710.workers.dev/events"

try:
    response = requests.get(WORKER_EVENTS_URL, timeout=10)
    response.raise_for_status()

    data = response.json()
    events = data.get("events", [])

    if len(events) == 0:
        st.warning("No live telemetry events found yet. Click the buttons on the Cloudflare website first.")
    else:
        df = pd.DataFrame(events)

        st.subheader("Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total events", len(df))

        with col2:
            if "risk_level" in df.columns:
                high_risk = df[df["risk_level"] == "high"].shape[0]
                st.metric("High-risk events", high_risk)
            else:
                st.metric("High-risk events", "N/A")

        with col3:
            if "risk_score" in df.columns:
                avg_risk_score = round(df["risk_score"].mean(), 2)
                st.metric("Average risk score", avg_risk_score)
            else:
                st.metric("Average risk score", "N/A")

        st.subheader("Prediction Counts")

        if "worker_prediction" in df.columns:
            prediction_counts = df["worker_prediction"].value_counts()
            st.bar_chart(prediction_counts)

        st.subheader("Risk Level Counts")

        if "risk_level" in df.columns:
            risk_counts = df["risk_level"].value_counts()
            st.bar_chart(risk_counts)

        st.subheader("Latest Live Telemetry Events")

        display_columns = [
            "id",
            "created_at",
            "page_path",
            "interaction_type",
            "user_agent_category",
            "worker_prediction",
            "risk_score",
            "risk_level",
            "action",
            "tls_version",
            "cipher_suite_count",
            "extension_count",
            "alpn",
            "sni_present"
        ]

        existing_columns = [col for col in display_columns if col in df.columns]

        st.dataframe(
            df[existing_columns],
            use_container_width=True
        )

        st.subheader("Raw Event Data")
        st.dataframe(df, use_container_width=True)

except requests.exceptions.RequestException as error:
    st.error("Could not load live telemetry data from the Cloudflare Worker.")
    st.write(error)