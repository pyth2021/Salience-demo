from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Salience Bot Detection Dashboard", layout="wide")

st.title("Salience Bot Detection Dashboard")

st.write(
    "Live dashboard for events collected from the Cloudflare Pages website, "
    "processed by the Cloudflare Worker, saved in D1, and displayed here."
)

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
        st.stop()

    df = pd.DataFrame(events)

    st.subheader("Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total events", len(df))

    with col2:
        high_risk = df[df.get("risk_level", "") == "high"].shape[0] if "risk_level" in df else "N/A"
        st.metric("High-risk events", high_risk)

    with col3:
        avg_risk = round(pd.to_numeric(df.get("risk_score", pd.Series(dtype=float)), errors="coerce").mean(), 2) if "risk_score" in df else "N/A"
        st.metric("Average risk score", avg_risk)

    with col4:
        unique_paths = df["page_path"].nunique() if "page_path" in df else "N/A"
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
        "sni_present",
    ]

    existing_columns = [column for column in preferred_columns if column in df.columns]
    st.dataframe(df[existing_columns], use_container_width=True)

    with st.expander("Raw Event Data"):
        st.dataframe(df, use_container_width=True)

except requests.exceptions.RequestException as error:
    st.error("Could not load live telemetry data from the Cloudflare Worker.")
    st.write(error)
