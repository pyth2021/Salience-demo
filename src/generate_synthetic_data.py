import csv
import os
import random
from datetime import datetime, timedelta

os.makedirs("data/raw", exist_ok=True)

output_file = "data/raw/synthetic_telemetry.csv"

fieldnames = [
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
    "label"
]


def random_timestamp(index):
    base_time = datetime.utcnow()
    return (base_time + timedelta(seconds=index * random.randint(1, 10))).isoformat()


def generate_human(index):
    return {
        "timestamp": random_timestamp(index),
        "page_path": random.choice(["/", "/products.html", "/contact.html", "/about.html"]),
        "interaction_type": random.choice(["normal_browsing", "button_click", "page_view"]),
        "scroll_depth_category": random.choice(["low", "medium", "high"]),
        "request_interval_seconds": round(random.uniform(3, 20), 2),
        "user_agent_category": "normal_browser",
        "has_favicon_request": 1,
        "requested_robots_txt": 0,
        "pages_per_session": random.randint(2, 10),
        "error_rate": round(random.uniform(0.0, 0.05), 2),
        "tls_version": "TLS1.3",
        "cipher_suite_count": random.randint(12, 18),
        "extension_count": random.randint(10, 16),
        "alpn": random.choice(["h2", "http/1.1"]),
        "sni_present": 1,
        "label": "human"
    }


def generate_good_bot(index):
    return {
        "timestamp": random_timestamp(index),
        "page_path": random.choice(["/robots.txt", "/sitemap.xml", "/products.html"]),
        "interaction_type": "crawler_request",
        "scroll_depth_category": "none",
        "request_interval_seconds": round(random.uniform(1.5, 6), 2),
        "user_agent_category": "good_bot",
        "has_favicon_request": random.choice([0, 1]),
        "requested_robots_txt": 1,
        "pages_per_session": random.randint(10, 40),
        "error_rate": round(random.uniform(0.0, 0.08), 2),
        "tls_version": "TLS1.3",
        "cipher_suite_count": random.randint(10, 16),
        "extension_count": random.randint(8, 14),
        "alpn": random.choice(["h2", "http/1.1"]),
        "sni_present": 1,
        "label": "good_bot"
    }


def generate_bad_bot(index):
    return {
        "timestamp": random_timestamp(index),
        "page_path": random.choice(["/products.html", "/login.html", "/checkout.html"]),
        "interaction_type": "rapid_request",
        "scroll_depth_category": "none",
        "request_interval_seconds": round(random.uniform(0.1, 1.0), 2),
        "user_agent_category": random.choice(["bot", "script", "python"]),
        "has_favicon_request": 0,
        "requested_robots_txt": 0,
        "pages_per_session": random.randint(60, 180),
        "error_rate": round(random.uniform(0.10, 0.35), 2),
        "tls_version": random.choice(["TLS1.2", "TLS1.3"]),
        "cipher_suite_count": random.randint(4, 9),
        "extension_count": random.randint(3, 8),
        "alpn": "http/1.1",
        "sni_present": random.choice([0, 1]),
        "label": "bad_bot"
    }


def generate_scanner(index):
    return {
        "timestamp": random_timestamp(index),
        "page_path": random.choice(["/admin", "/wp-login.php", "/phpmyadmin", "/.env", "/config"]),
        "interaction_type": "scanner_request",
        "scroll_depth_category": "none",
        "request_interval_seconds": round(random.uniform(0.05, 0.8), 2),
        "user_agent_category": random.choice(["curl", "python", "script"]),
        "has_favicon_request": 0,
        "requested_robots_txt": 0,
        "pages_per_session": random.randint(80, 250),
        "error_rate": round(random.uniform(0.30, 0.90), 2),
        "tls_version": random.choice(["TLS1.2", "TLS1.3"]),
        "cipher_suite_count": random.randint(3, 8),
        "extension_count": random.randint(2, 7),
        "alpn": "http/1.1",
        "sni_present": random.choice([0, 1]),
        "label": "scanner"
    }


rows = []

for i in range(250):
    rows.append(generate_human(i))

for i in range(250, 400):
    rows.append(generate_good_bot(i))

for i in range(400, 700):
    rows.append(generate_bad_bot(i))

for i in range(700, 1000):
    rows.append(generate_scanner(i))

random.shuffle(rows)

with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {len(rows)} rows in {output_file}")