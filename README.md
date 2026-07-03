# Salience Bot/Anomaly Detection Prototype

This repository contains a prototype for privacy-preserving bot and anomaly detection for website traffic. The project uses synthetic telemetry data to classify traffic as human, good bot, bad bot, or scanner-like activity.

The main goal is to test a full cloud-based detection flow, starting from a website beacon and ending with dashboard monitoring. The current prototype connects a Cloudflare Worker, an Azure-hosted ML API, a Cloudflare D1 database, and a Streamlit dashboard.

## Architecture

```text
Website / Beacon
        ↓
Cloudflare Worker
        ↓
Azure ML API
        ↓
Cloudflare D1 Database
        ↓
Streamlit Dashboard
```

## Current Progress

A synthetic dataset has been created to simulate different traffic types, including human visitors, good bots, bad bots, and scanner-like activity. The dataset includes behavioural and technical features such as request timing, pages per session, error rate, user-agent category, favicon and robots.txt behaviour, and TLS-inspired fields like TLS version, cipher suite count, extension count, ALPN, and SNI presence.

A rule-based detector has also been tested as a baseline. It uses simple scoring logic to flag suspicious patterns, such as very fast request timing, high error rates, high pages per session, suspicious user-agent categories, and scanner-like behaviour.

The machine learning API was built with Python and Flask. It loads a trained detection model and provides endpoints for health checks and predictions. The API has been tested locally and deployed using Azure App Service.

The Cloudflare Worker acts as the edge layer. It receives telemetry from the website beacon, sends the feature data to the Azure ML API, receives the prediction, and stores the result in Cloudflare D1.

The Streamlit dashboard is used to view recent events, prediction results, risk levels, and summary information from the database.

## Main Components

| Component | Purpose |
|---|---|
| Website / Beacon | Collects minimized telemetry from website visits |
| Cloudflare Worker | Processes telemetry and connects the website to the ML API |
| Azure ML API | Hosts the trained detection model and returns predictions |
| Cloudflare D1 | Stores telemetry events and prediction results |
| Streamlit Dashboard | Displays detection results and recent activity |
| Synthetic Dataset | Used for training and testing detection models |

## Privacy Design

The prototype follows a privacy-minimizing approach. It is designed to avoid intentionally storing sensitive personal information such as passwords, cookies, authentication tokens, raw IP addresses, names, emails, or private user content.

The system focuses only on behavioural and technical signals needed for bot and anomaly detection.

## Current Goal

The current goal is to make the prototype more realistic by connecting the full flow to a simple test website. This will help confirm that real visit telemetry can move through the pipeline.

```text
Website → Cloudflare Worker → Azure ML API → Cloudflare D1 → Streamlit Dashboard
```

## Next Steps

The next tasks are to build or deploy a simple test website, connect the beacon to the existing Worker, test real visit data through the full system, measure end-to-end latency, save screenshots for project evidence, and finalize the implementation, evaluation, and privacy/compliance sections for the final report.

## Notes

This project is still in progress. The current dataset is synthetic and is mainly used for prototype testing, model development, and validating the end-to-end architecture.
