import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Salience Bot Detection Dashboard",
    layout="wide"
)

st.title("Salience Bot Detection Dashboard")

st.write(
    "This dashboard shows telemetry events collected from the static website beacon "
    "and classified by the ML bot detection model."
)

DATA_FILE = "data/raw/telemetry_log.csv"

try:
    df = pd.read_csv(DATA_FILE)

    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total events", len(df))

    with col2:
        if "ml_prediction" in df.columns:
            high_risk = df[df["ml_prediction"].isin(["bad_bot", "scanner"])].shape[0]
            st.metric("High-risk events", high_risk)
        else:
            st.metric("High-risk events", "N/A")

    with col3:
        if "confidence" in df.columns:
            avg_confidence = round(df["confidence"].mean(), 3)
            st.metric("Average confidence", avg_confidence)
        else:
            st.metric("Average confidence", "N/A")

    st.subheader("Prediction Counts")

    if "ml_prediction" in df.columns:
        prediction_counts = df["ml_prediction"].value_counts()
        st.bar_chart(prediction_counts)

    st.subheader("Risk Level Counts")

    if "risk_level" in df.columns:
        risk_counts = df["risk_level"].value_counts()
        st.bar_chart(risk_counts)

    st.subheader("Latest Telemetry Events")

    st.dataframe(df.tail(20), use_container_width=True)

except FileNotFoundError:
    st.error("No telemetry_log.csv file found yet. Click the website buttons first to generate telemetry.")