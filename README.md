# Privacy-Preserving Bot & Anomaly Detection for Static-Site Telemetry

A Cybersecurity and Artificial Intelligence capstone implementation for **Salience Enterprises Inc.** This prototype demonstrates privacy-aware telemetry collection, bot/anomaly classification, risk scoring, Cloudflare D1 storage, and Streamlit dashboard reporting for a static website.

## Project Identity

- **Institution:** Humber Polytechnic
- **Faculty:** Faculty of Applied Sciences & Technology
- **Program:** Cybersecurity and Artificial Intelligence
- **Project Sponsor:** Salience Enterprises Inc.
- **Industry Supervisor:** Abdullah Ali Syed
- **Course Instructor:** Asama Nseaf

## Team Members

1. Mahin Chowdhury
2. Naveen Saranya Patro Behara
3. Olamide Akinyosola
4. Suma Madasu

## Live Links

- **Website:** https://salience-demo.pages.dev
- **Worker Events API:** https://salience-beacon-worker.mahin0710.workers.dev/events
- **Dashboard:** https://salience-demo-jtz9t5jqcu5c2scubyrt7r.streamlit.app/
- **Repository:** https://github.com/pyth2021/Salience-demo

## Website Pages

The live website is structured as a multi-page Cloudflare Pages site:

- `index.html` — project overview, privacy boundaries, team, and live project links
- `pipeline.html` — architecture, system ownership, detection methods, and scope boundaries
- `live-stats.html` — controlled telemetry test buttons, dashboard link, Worker API link, and model evaluation summary

## Implemented Flow

```text
Static website → beacon.js → Cloudflare Worker → privacy validation → feature extraction → Azure /predict ML API → Cloudflare D1 → Streamlit Dashboard
```

## Detection Models

The project includes two trained pickle model files in both `models/` and `ml_api/models/`:

- `isolation_forest_model.pkl` — unsupervised anomaly detection model
- `gradient_boosting_model.pkl` — supervised classification model

A compatibility file named `bot_detection_model.pkl` is also kept for older deployment settings that may still reference the previous model filename.

## Prototype Scope

The prototype focuses on:

- Limited telemetry collection from a static website
- Privacy validation and minimized telemetry schema
- Rule-based detection and ML-assisted classification
- Synthetic dataset and feature engineering pipeline
- Cloudflare Worker intake endpoint
- Azure `/predict` ML API integration
- Cloudflare D1 event storage
- Streamlit Dashboard reporting
- Model evaluation using precision, recall, F1-score, false positive rate, and confusion matrix evidence

The prototype does **not** claim production-grade blocking, CAPTCHA enforcement, rate limiting, active mitigation, third-party attack testing, or commercial-ready bot-management capability.

## Privacy Boundary

The telemetry design avoids raw IP addresses, names, emails, passwords, cookies, authentication tokens, private user content, exact location tracking, and invasive browser fingerprinting.

## Train Models

From the project root:

```bash
python src/train_model.py
```

This regenerates:

- `models/gradient_boosting_model.pkl`
- `models/isolation_forest_model.pkl`
- `ml_api/models/gradient_boosting_model.pkl`
- `ml_api/models/isolation_forest_model.pkl`
- `dashboard/dashboard/model_evaluation.json`
