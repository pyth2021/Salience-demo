# Privacy-Preserving Bot & Anomaly Detection for Static-Site Telemetry

A Cybersecurity and Artificial Intelligence capstone implementation for **Salience Enterprises Inc.** This prototype demonstrates privacy-aware telemetry collection, bot/anomaly classification, risk scoring, Cloudflare D1 storage, and Streamlit dashboard reporting for a static website.

## Project Identity

- **Institution:** Humber Polytechnic, Toronto, Ontario
- **Faculty:** Faculty of Applied Sciences & Technology
- **Program:** Cybersecurity and Artificial Intelligence
- **Project Sponsor:** Salience Enterprises Inc.
- **Industry Supervisor:** Abdullah Ali Syed
- **Course Instructor:** Asama Nseaf

## Team Members

Mahin Chowdhury
Naveen Saranya Patro Behara
Olamide Akinyosola
Suma Madasu

## Live Links

- **Website:** https://salience-demo.pages.dev
- **Worker Events API:** https://salience-beacon-worker.mahin0710.workers.dev/events
- **Dashboard:** https://salience-demo-we5wxbfsagmuuzj3vyjvog.streamlit.app
- **Repository:** https://github.com/pyth2021/Salience-demo

## Website Pages

The live website is structured as a multi-page Cloudflare Pages site:

- `index.html` — project overview, privacy boundaries, team, and live project links
- `pipeline.html` — architecture, system ownership, detection methods, and scope boundaries
- `live-stats.html` — controlled telemetry test buttons, dashboard link, Worker API link, and model evaluation summary

## Implemented Flow

```text
Static website → beacon.js → Cloudflare Worker → privacy validation → feature extraction → Azure /predict ML API → Cloudflare D1 → Streamlit Dashboard