from __future__ import annotations

import csv
import random
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = OUTPUT_DIRECTORY / "synthetic_train.csv"
TEST_FILE = OUTPUT_DIRECTORY / "synthetic_test.csv"


# =============================================================================
# DATASET SETTINGS
# =============================================================================

TRAIN_RANDOM_SEED = 42
TEST_RANDOM_SEED = 4242


# Balanced development data for Gradient Boosting.
TRAIN_CLASS_COUNTS = {
    "human": 10_000,
    "good_bot": 10_000,
    "bad_bot": 10_000,
    "scanner": 10_000,
}


# More realistic operational distribution for final evaluation.
TEST_CLASS_COUNTS = {
    "human": 7_000,
    "good_bot": 1_500,
    "bad_bot": 1_000,
    "scanner": 500,
}


# These are the exact 14 model-input features.
MODEL_FEATURES = [
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
]


# Training data contains:
# timestamp + 14 model features + supervised target label.
TRAIN_FIELDNAMES = [
    "timestamp",
    *MODEL_FEATURES,
    "label",
]


# Final holdout data also contains anomaly ground truth.
# This field is for Isolation Forest evaluation only.
TEST_FIELDNAMES = [
    "timestamp",
    *MODEL_FEATURES,
    "label",
    "anomaly_ground_truth",
]


# =============================================================================
# BASE TRAFFIC PROFILES
# =============================================================================
#
# All classes share the same possible category values.
# Their probabilities overlap so no single feature reveals the answer.
#
# These values are synthetic project assumptions, not verified production
# thresholds.
# =============================================================================

BASE_PROFILES: dict[str, dict[str, Any]] = {
    "human": {
        "interval": (7.0, 4.0, 0.25, 25.0),
        "pages": (8.0, 5.0, 1, 32),
        "error": (0.025, 0.030, 0.0, 0.20),

        "page_category": {
            "public_page": 0.58,
            "account_page": 0.17,
            "checkout_page": 0.10,
            "crawler_file": 0.01,
            "sensitive_page": 0.02,
            "unknown_page": 0.12,
        },

        "interaction_type": {
            "page_view": 0.32,
            "navigation": 0.31,
            "resource_request": 0.13,
            "form_request": 0.14,
            "api_request": 0.06,
            "automated_request": 0.04,
        },

        "scroll_depth": {
            "none": 0.08,
            "low": 0.25,
            "medium": 0.42,
            "high": 0.25,
        },

        "user_agent": {
            "browser": 0.88,
            "crawler": 0.01,
            "script_client": 0.03,
            "command_line": 0.01,
            "unknown": 0.07,
        },

        "favicon_probability": 0.90,
        "robots_probability": 0.01,
        "tls13_probability": 0.88,
        "h2_probability": 0.82,
        "sni_probability": 0.995,
    },

    "good_bot": {
        "interval": (3.2, 2.4, 0.20, 14.0),
        "pages": (28.0, 18.0, 2, 100),
        "error": (0.045, 0.045, 0.0, 0.28),

        "page_category": {
            "public_page": 0.42,
            "account_page": 0.05,
            "checkout_page": 0.02,
            "crawler_file": 0.34,
            "sensitive_page": 0.03,
            "unknown_page": 0.14,
        },

        "interaction_type": {
            "page_view": 0.06,
            "navigation": 0.08,
            "resource_request": 0.29,
            "form_request": 0.01,
            "api_request": 0.18,
            "automated_request": 0.38,
        },

        "scroll_depth": {
            "none": 0.84,
            "low": 0.11,
            "medium": 0.04,
            "high": 0.01,
        },

        "user_agent": {
            "browser": 0.08,
            "crawler": 0.68,
            "script_client": 0.10,
            "command_line": 0.03,
            "unknown": 0.11,
        },

        # Base favicon chance is low for backend crawlers.
        # Individual subtypes adjust this value.
        "favicon_probability": 0.10,
        "robots_probability": 0.72,
        "tls13_probability": 0.84,
        "h2_probability": 0.76,
        "sni_probability": 0.992,
    },

    "bad_bot": {
        "interval": (1.5, 1.4, 0.05, 8.0),
        "pages": (58.0, 35.0, 5, 220),
        "error": (0.150, 0.120, 0.0, 0.70),

        "page_category": {
            "public_page": 0.40,
            "account_page": 0.18,
            "checkout_page": 0.10,
            "crawler_file": 0.04,
            "sensitive_page": 0.10,
            "unknown_page": 0.18,
        },

        "interaction_type": {
            "page_view": 0.10,
            "navigation": 0.10,
            "resource_request": 0.19,
            "form_request": 0.13,
            "api_request": 0.16,
            "automated_request": 0.32,
        },

        "scroll_depth": {
            "none": 0.62,
            "low": 0.22,
            "medium": 0.12,
            "high": 0.04,
        },

        "user_agent": {
            "browser": 0.34,
            "crawler": 0.10,
            "script_client": 0.23,
            "command_line": 0.09,
            "unknown": 0.24,
        },

        "favicon_probability": 0.38,
        "robots_probability": 0.10,
        "tls13_probability": 0.73,
        "h2_probability": 0.61,
        "sni_probability": 0.965,
    },

    "scanner": {
        "interval": (1.0, 1.2, 0.03, 8.0),
        "pages": (90.0, 55.0, 8, 280),
        "error": (0.440, 0.220, 0.04, 0.98),

        "page_category": {
            "public_page": 0.09,
            "account_page": 0.13,
            "checkout_page": 0.03,
            "crawler_file": 0.02,
            "sensitive_page": 0.50,
            "unknown_page": 0.23,
        },

        "interaction_type": {
            "page_view": 0.03,
            "navigation": 0.04,
            "resource_request": 0.11,
            "form_request": 0.05,
            "api_request": 0.19,
            "automated_request": 0.58,
        },

        "scroll_depth": {
            "none": 0.88,
            "low": 0.08,
            "medium": 0.03,
            "high": 0.01,
        },

        "user_agent": {
            "browser": 0.15,
            "crawler": 0.06,
            "script_client": 0.27,
            "command_line": 0.23,
            "unknown": 0.29,
        },

        "favicon_probability": 0.18,
        "robots_probability": 0.03,
        "tls13_probability": 0.67,
        "h2_probability": 0.49,
        "sni_probability": 0.935,
    },
}


# =============================================================================
# HIDDEN BEHAVIOURAL SUBTYPES
# =============================================================================
#
# Subtypes create variety inside each class.
# They are never saved to the CSV and are never provided to the model.
# =============================================================================

SUBTYPES: dict[str, dict[str, dict[str, Any]]] = {
    "human": {
        "casual_reader": {
            "interval_multiplier": 1.25,
            "pages_multiplier": 0.70,
            "anomaly_probability": 0.01,
        },

        "engaged_visitor": {
            "interval_multiplier": 0.80,
            "pages_multiplier": 1.55,
            "anomaly_probability": 0.01,

            "scroll_depth": {
                "none": 0.03,
                "low": 0.12,
                "medium": 0.40,
                "high": 0.45,
            },
        },

        "fast_navigator": {
            "interval_multiplier": 0.35,
            "pages_multiplier": 1.45,
            "favicon_adjustment": -0.06,
            "anomaly_probability": 0.05,
        },

        "privacy_focused_human": {
            "interval_multiplier": 0.85,
            "pages_multiplier": 0.90,
            "error_adjustment": 0.025,
            "favicon_adjustment": -0.52,
            "tls13_adjustment": -0.08,
            "h2_adjustment": -0.18,
            "sni_adjustment": -0.010,
            "anomaly_probability": 0.08,

            "user_agent": {
                "browser": 0.58,
                "crawler": 0.03,
                "script_client": 0.08,
                "command_line": 0.04,
                "unknown": 0.27,
            },

            "scroll_depth": {
                "none": 0.16,
                "low": 0.31,
                "medium": 0.36,
                "high": 0.17,
            },
        },

        "unusual_burst": {
            "interval_multiplier": 0.16,
            "pages_multiplier": 3.00,
            "error_adjustment": 0.10,
            "anomaly_probability": 0.50,

            "user_agent": {
                "browser": 0.62,
                "crawler": 0.04,
                "script_client": 0.11,
                "command_line": 0.04,
                "unknown": 0.19,
            },
        },
    },

    "good_bot": {
        "search_crawler": {
            "pages_multiplier": 1.25,
            "favicon_adjustment": -0.04,
            "robots_adjustment": 0.14,
            "anomaly_probability": 0.03,
        },

        "sitemap_crawler": {
            "interval_multiplier": 1.35,
            "favicon_adjustment": -0.07,
            "robots_adjustment": 0.18,
            "anomaly_probability": 0.02,

            "page_category": {
                "public_page": 0.29,
                "account_page": 0.03,
                "checkout_page": 0.01,
                "crawler_file": 0.51,
                "sensitive_page": 0.02,
                "unknown_page": 0.14,
            },
        },

        # Monitoring bots may use a headless browser and request favicons.
        "monitoring_bot": {
            "interval_multiplier": 2.20,
            "pages_multiplier": 0.40,
            "favicon_adjustment": 0.25,
            "robots_adjustment": -0.20,
            "anomaly_probability": 0.07,

            "user_agent": {
                "browser": 0.28,
                "crawler": 0.30,
                "script_client": 0.25,
                "command_line": 0.05,
                "unknown": 0.12,
            },
        },

        "misconfigured_crawler": {
            "interval_multiplier": 0.33,
            "pages_multiplier": 2.10,
            "error_adjustment": 0.16,
            "favicon_adjustment": 0.10,
            "robots_adjustment": -0.30,
            "anomaly_probability": 0.50,
        },
    },

    "bad_bot": {
        "stealth_scraper": {
            "interval_multiplier": 1.35,
            "pages_multiplier": 0.80,
            "error_adjustment": -0.05,
            "favicon_adjustment": 0.25,
            "anomaly_probability": 0.88,

            "user_agent": {
                "browser": 0.54,
                "crawler": 0.12,
                "script_client": 0.15,
                "command_line": 0.04,
                "unknown": 0.15,
            },
        },

        "aggressive_scraper": {
            "interval_multiplier": 0.28,
            "pages_multiplier": 1.90,
            "error_adjustment": 0.11,
            "anomaly_probability": 0.99,
        },

        "credential_automation": {
            "interval_multiplier": 0.55,
            "error_adjustment": 0.14,
            "anomaly_probability": 0.98,

            "page_category": {
                "public_page": 0.08,
                "account_page": 0.47,
                "checkout_page": 0.20,
                "crawler_file": 0.01,
                "sensitive_page": 0.13,
                "unknown_page": 0.11,
            },

            "interaction_type": {
                "page_view": 0.03,
                "navigation": 0.05,
                "resource_request": 0.09,
                "form_request": 0.45,
                "api_request": 0.18,
                "automated_request": 0.20,
            },
        },

        # This subtype intentionally resembles normal browser traffic.
        "fake_browser_bot": {
            "interval_multiplier": 2.40,
            "pages_multiplier": 0.55,
            "error_adjustment": -0.10,
            "favicon_adjustment": 0.45,
            "tls13_adjustment": 0.10,
            "h2_adjustment": 0.15,
            "sni_adjustment": 0.020,
            "anomaly_probability": 0.84,

            "page_category": {
                "public_page": 0.52,
                "account_page": 0.19,
                "checkout_page": 0.12,
                "crawler_file": 0.02,
                "sensitive_page": 0.04,
                "unknown_page": 0.11,
            },

            "interaction_type": {
                "page_view": 0.23,
                "navigation": 0.24,
                "resource_request": 0.16,
                "form_request": 0.13,
                "api_request": 0.08,
                "automated_request": 0.16,
            },

            "scroll_depth": {
                "none": 0.24,
                "low": 0.31,
                "medium": 0.30,
                "high": 0.15,
            },

            "user_agent": {
                "browser": 0.72,
                "crawler": 0.04,
                "script_client": 0.08,
                "command_line": 0.02,
                "unknown": 0.14,
            },
        },
    },

    "scanner": {
        "broad_scanner": {
            "interval_multiplier": 0.30,
            "pages_multiplier": 1.75,
            "error_adjustment": 0.18,
            "anomaly_probability": 0.998,
        },

        "targeted_scanner": {
            "pages_multiplier": 0.72,
            "anomaly_probability": 0.99,

            "page_category": {
                "public_page": 0.06,
                "account_page": 0.18,
                "checkout_page": 0.04,
                "crawler_file": 0.01,
                "sensitive_page": 0.58,
                "unknown_page": 0.13,
            },
        },

        "slow_scanner": {
            "interval_multiplier": 3.50,
            "pages_multiplier": 0.45,
            "error_adjustment": -0.08,
            "anomaly_probability": 0.96,

            "user_agent": {
                "browser": 0.19,
                "crawler": 0.07,
                "script_client": 0.22,
                "command_line": 0.17,
                "unknown": 0.35,
            },
        },

        "evasive_scanner": {
            "interval_multiplier": 2.10,
            "pages_multiplier": 0.58,
            "error_adjustment": -0.14,
            "favicon_adjustment": 0.24,
            "tls13_adjustment": 0.08,
            "h2_adjustment": 0.12,
            "anomaly_probability": 0.98,

            "user_agent": {
                "browser": 0.34,
                "crawler": 0.07,
                "script_client": 0.20,
                "command_line": 0.11,
                "unknown": 0.28,
            },
        },
    },
}


# =============================================================================
# SUBTYPE DISTRIBUTIONS
# =============================================================================

SUBTYPE_WEIGHTS = {
    "train": {
        "human": {
            "casual_reader": 0.38,
            "engaged_visitor": 0.29,
            "fast_navigator": 0.17,
            "privacy_focused_human": 0.11,
            "unusual_burst": 0.05,
        },

        "good_bot": {
            "search_crawler": 0.48,
            "sitemap_crawler": 0.25,
            "monitoring_bot": 0.20,
            "misconfigured_crawler": 0.07,
        },

        "bad_bot": {
            "stealth_scraper": 0.34,
            "aggressive_scraper": 0.28,
            "credential_automation": 0.22,
            "fake_browser_bot": 0.16,
        },

        "scanner": {
            "broad_scanner": 0.38,
            "targeted_scanner": 0.27,
            "slow_scanner": 0.20,
            "evasive_scanner": 0.15,
        },
    },

    "test": {
        "human": {
            "casual_reader": 0.33,
            "engaged_visitor": 0.27,
            "fast_navigator": 0.23,
            "privacy_focused_human": 0.13,
            "unusual_burst": 0.04,
        },

        "good_bot": {
            "search_crawler": 0.42,
            "sitemap_crawler": 0.20,
            "monitoring_bot": 0.28,
            "misconfigured_crawler": 0.10,
        },

        "bad_bot": {
            "stealth_scraper": 0.41,
            "aggressive_scraper": 0.18,
            "credential_automation": 0.20,
            "fake_browser_bot": 0.21,
        },

        "scanner": {
            "broad_scanner": 0.23,
            "targeted_scanner": 0.25,
            "slow_scanner": 0.27,
            "evasive_scanner": 0.25,
        },
    },
}


# =============================================================================
# GENERAL HELPER FUNCTIONS
# =============================================================================

def weighted_choice(
    rng: random.Random,
    probabilities: dict[str, float],
) -> str:
    return rng.choices(
        population=list(probabilities.keys()),
        weights=list(probabilities.values()),
        k=1,
    )[0]


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(maximum, value),
    )


def bounded_float(
    rng: random.Random,
    mean: float,
    standard_deviation: float,
    minimum: float,
    maximum: float,
    decimal_places: int,
) -> float:
    value = rng.gauss(
        mean,
        standard_deviation,
    )

    return round(
        clamp(value, minimum, maximum),
        decimal_places,
    )


def bounded_integer(
    rng: random.Random,
    mean: float,
    standard_deviation: float,
    minimum: int,
    maximum: int,
) -> int:
    value = round(
        rng.gauss(
            mean,
            standard_deviation,
        )
    )

    return int(
        clamp(value, minimum, maximum)
    )


def probability_flag(
    rng: random.Random,
    probability: float,
) -> int:
    probability = clamp(
        probability,
        0.0,
        1.0,
    )

    return int(
        rng.random() < probability
    )


def create_timestamp(
    rng: random.Random,
    split: str,
    index: int,
) -> str:
    """
    Generate reproducible synthetic timestamps.

    The holdout set uses a later synthetic time window.
    Timestamp remains metadata and is not a model feature.
    """

    if split == "train":
        start_time = datetime(
            2025,
            1,
            1,
            tzinfo=timezone.utc,
        )

        number_of_days = 180

    else:
        start_time = datetime(
            2025,
            7,
            1,
            tzinfo=timezone.utc,
        )

        number_of_days = 90

    random_seconds = rng.randint(
        0,
        number_of_days * 24 * 60 * 60,
    )

    timestamp = start_time + timedelta(
        seconds=random_seconds,
        microseconds=index,
    )

    return timestamp.isoformat()


# =============================================================================
# CREATE A COMPLETE SUBTYPE PROFILE
# =============================================================================

def create_profile(
    label: str,
    subtype: str,
) -> dict[str, Any]:
    profile = deepcopy(
        BASE_PROFILES[label]
    )

    adjustments = SUBTYPES[label][subtype]

    profile["interval_multiplier"] = adjustments.get(
        "interval_multiplier",
        1.0,
    )

    profile["pages_multiplier"] = adjustments.get(
        "pages_multiplier",
        1.0,
    )

    profile["error_adjustment"] = adjustments.get(
        "error_adjustment",
        0.0,
    )

    profile["favicon_probability"] += adjustments.get(
        "favicon_adjustment",
        0.0,
    )

    profile["robots_probability"] += adjustments.get(
        "robots_adjustment",
        0.0,
    )

    profile["tls13_probability"] += adjustments.get(
        "tls13_adjustment",
        0.0,
    )

    profile["h2_probability"] += adjustments.get(
        "h2_adjustment",
        0.0,
    )

    profile["sni_probability"] += adjustments.get(
        "sni_adjustment",
        0.0,
    )

    profile["anomaly_probability"] = adjustments[
        "anomaly_probability"
    ]

    probability_fields = [
        "favicon_probability",
        "robots_probability",
        "tls13_probability",
        "h2_probability",
        "sni_probability",
    ]

    for field in probability_fields:
        profile[field] = clamp(
            profile[field],
            0.0,
            1.0,
        )

    override_fields = [
        "page_category",
        "interaction_type",
        "scroll_depth",
        "user_agent",
    ]

    for field in override_fields:
        if field in adjustments:
            profile[field] = adjustments[field]

    return profile


# =============================================================================
# CORRELATED TLS-STYLE TECHNICAL FEATURES
# =============================================================================

def create_technical_counts(
    rng: random.Random,
    user_agent_category: str,
    tls_version: str,
    split: str,
) -> tuple[int, int]:
    """
    Generate cipher-suite and extension counts.

    These counts depend on the coarse client category and TLS version,
    rather than directly depending on the target label.
    """

    technical_profiles = {
        "browser": {
            "cipher": (14, 2.0, 7, 20),
            "extension": (12, 2.0, 6, 17),
        },

        "crawler": {
            "cipher": (12, 2.4, 6, 19),
            "extension": (10, 2.3, 5, 16),
        },

        "script_client": {
            "cipher": (8.5, 2.5, 3, 16),
            "extension": (7, 2.4, 2, 14),
        },

        "command_line": {
            "cipher": (6, 2.0, 2, 13),
            "extension": (5, 1.8, 1, 11),
        },

        "unknown": {
            "cipher": (9.5, 3.0, 2, 18),
            "extension": (8, 3.0, 1, 16),
        },
    }

    settings = technical_profiles[
        user_agent_category
    ]

    if split == "test":
        distribution_shift = rng.uniform(
            -0.8,
            0.8,
        )
    else:
        distribution_shift = 0.0

    (
        cipher_mean,
        cipher_standard_deviation,
        cipher_minimum,
        cipher_maximum,
    ) = settings["cipher"]

    (
        extension_mean,
        extension_standard_deviation,
        extension_minimum,
        extension_maximum,
    ) = settings["extension"]

    cipher_count = bounded_integer(
        rng,
        cipher_mean + distribution_shift,
        cipher_standard_deviation,
        cipher_minimum,
        cipher_maximum,
    )

    extension_count = bounded_integer(
        rng,
        extension_mean + distribution_shift,
        extension_standard_deviation,
        extension_minimum,
        extension_maximum,
    )

    # Modern TLS clients may advertise slightly richer profiles.
    if tls_version == "TLS1.3":
        cipher_count = min(
            20,
            cipher_count
            + probability_flag(rng, 0.35),
        )

        extension_count = min(
            17,
            extension_count
            + probability_flag(rng, 0.40),
        )

    return (
        cipher_count,
        extension_count,
    )


# =============================================================================
# CORRELATED ALPN GENERATION
# =============================================================================

def select_alpn(
    rng: random.Random,
    base_h2_probability: float,
    tls_version: str,
    user_agent_category: str,
    extension_count: int,
) -> str:
    """
    Generate ALPN based on related network characteristics.

    HTTP/2 remains possible with TLS 1.2 and TLS 1.3, but is more common
    for modern browser-like TLS profiles.
    """

    h2_probability = base_h2_probability

    # Modern TLS increases the probability of HTTP/2.
    if tls_version == "TLS1.3":
        h2_probability += 0.05
    else:
        h2_probability -= 0.05

    # Browser clients are more likely to negotiate HTTP/2.
    if user_agent_category == "browser":
        h2_probability += 0.05

    # Simple scripts and command-line clients often use HTTP/1.1.
    elif user_agent_category in {
        "script_client",
        "command_line",
    }:
        h2_probability -= 0.15

    # Very small extension sets are less representative of an h2 browser.
    if extension_count <= 4:
        h2_probability -= 0.15

    elif extension_count >= 11:
        h2_probability += 0.04

    h2_probability = clamp(
        h2_probability,
        0.02,
        0.97,
    )

    if rng.random() < h2_probability:
        return "h2"

    return "http/1.1"


# =============================================================================
# GENERATE ONE SYNTHETIC ROW
# =============================================================================

def generate_row(
    rng: random.Random,
    label: str,
    subtype: str,
    split: str,
    index: int,
) -> dict[str, Any]:
    profile = create_profile(
        label,
        subtype,
    )

    (
        interval_mean,
        interval_standard_deviation,
        interval_minimum,
        interval_maximum,
    ) = profile["interval"]

    (
        pages_mean,
        pages_standard_deviation,
        pages_minimum,
        pages_maximum,
    ) = profile["pages"]

    (
        error_mean,
        error_standard_deviation,
        error_minimum,
        error_maximum,
    ) = profile["error"]

    interval_mean *= profile[
        "interval_multiplier"
    ]

    pages_mean *= profile[
        "pages_multiplier"
    ]

    error_mean += profile[
        "error_adjustment"
    ]

    # The final test set has a mild distribution shift.
    if split == "test":
        interval_mean *= rng.uniform(
            0.85,
            1.18,
        )

        pages_mean *= rng.uniform(
            0.88,
            1.15,
        )

        error_mean += rng.uniform(
            -0.015,
            0.035,
        )

    request_interval = bounded_float(
        rng,
        interval_mean,
        interval_standard_deviation,
        max(0.03, interval_minimum),
        interval_maximum,
        2,
    )

    # Faster request cadence usually creates more session activity.
    pace_factor = clamp(
        interval_mean
        / max(request_interval, 0.03),
        0.60,
        1.80,
    )

    correlated_pages_mean = (
        pages_mean
        * (0.72 + 0.28 * pace_factor)
    )

    pages_per_session = bounded_integer(
        rng,
        correlated_pages_mean,
        pages_standard_deviation,
        pages_minimum,
        pages_maximum,
    )

    page_category = weighted_choice(
        rng,
        profile["page_category"],
    )

    interaction_type = weighted_choice(
        rng,
        profile["interaction_type"],
    )

    scroll_depth_category = weighted_choice(
        rng,
        profile["scroll_depth"],
    )

    user_agent_category = weighted_choice(
        rng,
        profile["user_agent"],
    )

    # Automated requests usually have less scrolling.
    automated_interaction_types = {
        "automated_request",
        "resource_request",
        "api_request",
    }

    if (
        interaction_type in automated_interaction_types
        and rng.random() < 0.68
    ):
        scroll_depth_category = weighted_choice(
            rng,
            {
                "none": 0.82,
                "low": 0.13,
                "medium": 0.04,
                "high": 0.01,
            },
        )

    if rng.random() < profile[
        "tls13_probability"
    ]:
        tls_version = "TLS1.3"
    else:
        tls_version = "TLS1.2"

    (
        cipher_suite_count,
        extension_count,
    ) = create_technical_counts(
        rng,
        user_agent_category,
        tls_version,
        split,
    )

    # ALPN now depends on TLS, client category, and extension count.
    alpn = select_alpn(
        rng=rng,
        base_h2_probability=profile[
            "h2_probability"
        ],
        tls_version=tls_version,
        user_agent_category=user_agent_category,
        extension_count=extension_count,
    )

    # Coarse page category affects the expected error rate.
    page_error_adjustment = {
        "public_page": -0.010,
        "account_page": 0.015,
        "checkout_page": 0.020,
        "crawler_file": -0.005,
        "sensitive_page": 0.090,
        "unknown_page": 0.060,
    }[page_category]

    # Extremely fast requests may produce more failures.
    if request_interval < 0.25:
        cadence_error_adjustment = 0.045

    elif request_interval < 0.75:
        cadence_error_adjustment = 0.020

    else:
        cadence_error_adjustment = 0.0

    error_rate = bounded_float(
        rng,
        (
            error_mean
            + page_error_adjustment
            + cadence_error_adjustment
        ),
        error_standard_deviation,
        error_minimum,
        error_maximum,
        3,
    )

    favicon_probability = profile[
        "favicon_probability"
    ]

    # Browser-style clients are more likely to request favicons.
    if user_agent_category == "browser":
        favicon_probability += 0.07

    elif user_agent_category in {
        "script_client",
        "command_line",
    }:
        favicon_probability -= 0.08

    favicon_probability = clamp(
        favicon_probability,
        0.01,
        0.98,
    )

    robots_probability = profile[
        "robots_probability"
    ]

    if page_category == "crawler_file":
        robots_probability += 0.08

    robots_probability = clamp(
        robots_probability,
        0.0,
        0.98,
    )

    row = {
        "timestamp": create_timestamp(
            rng,
            split,
            index,
        ),

        "page_category": page_category,

        "interaction_type": interaction_type,

        "scroll_depth_category": (
            scroll_depth_category
        ),

        "request_interval_seconds": (
            request_interval
        ),

        "user_agent_category": (
            user_agent_category
        ),

        "has_favicon_request": probability_flag(
            rng,
            favicon_probability,
        ),

        "requested_robots_txt": probability_flag(
            rng,
            robots_probability,
        ),

        "pages_per_session": pages_per_session,

        "error_rate": error_rate,

        "tls_version": tls_version,

        "cipher_suite_count": (
            cipher_suite_count
        ),

        "extension_count": extension_count,

        "alpn": alpn,

        "sni_present": probability_flag(
            rng,
            profile["sni_probability"],
        ),

        # Supervised Gradient Boosting ground truth.
        "label": label,
    }

    # Evaluation-only field.
    # It is included only in the final holdout file.
    if split == "test":
        row["anomaly_ground_truth"] = (
            probability_flag(
                rng,
                profile["anomaly_probability"],
            )
        )

    return row


# =============================================================================
# GENERATE A COMPLETE DATASET
# =============================================================================

def generate_dataset(
    split: str,
    class_counts: dict[str, int],
    random_seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(
        random_seed
    )

    rows: list[dict[str, Any]] = []

    row_index = 0

    for label, number_of_rows in class_counts.items():
        subtype_probabilities = (
            SUBTYPE_WEIGHTS[split][label]
        )

        for _ in range(number_of_rows):
            subtype = weighted_choice(
                rng,
                subtype_probabilities,
            )

            row = generate_row(
                rng=rng,
                label=label,
                subtype=subtype,
                split=split,
                index=row_index,
            )

            rows.append(row)
            row_index += 1

    rng.shuffle(rows)

    return rows


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_dataset(
    rows: list[dict[str, Any]],
    expected_class_counts: dict[str, int],
    expected_fieldnames: list[str],
    dataset_name: str,
) -> None:
    expected_total = sum(
        expected_class_counts.values()
    )

    if len(rows) != expected_total:
        raise ValueError(
            f"{dataset_name}: expected "
            f"{expected_total} rows, but generated "
            f"{len(rows)}."
        )

    actual_class_counts = Counter(
        row["label"]
        for row in rows
    )

    if actual_class_counts != Counter(
        expected_class_counts
    ):
        raise ValueError(
            f"{dataset_name}: incorrect class "
            f"counts: {dict(actual_class_counts)}"
        )

    for row in rows:
        if list(row.keys()) != expected_fieldnames:
            raise ValueError(
                f"{dataset_name}: row schema does "
                f"not match the expected fields."
            )

        if float(
            row["request_interval_seconds"]
        ) <= 0:
            raise ValueError(
                f"{dataset_name}: request interval "
                f"must be positive."
            )

        if not 0.0 <= float(
            row["error_rate"]
        ) <= 1.0:
            raise ValueError(
                f"{dataset_name}: error_rate must "
                f"be between 0 and 1."
            )

        if (
            "anomaly_ground_truth" in row
            and row["anomaly_ground_truth"]
            not in {0, 1}
        ):
            raise ValueError(
                f"{dataset_name}: "
                f"anomaly_ground_truth must be "
                f"either 0 or 1."
            )


# =============================================================================
# WRITE CSV FILE
# =============================================================================

def write_csv(
    rows: list[dict[str, Any]],
    output_file: Path,
    fieldnames: list[str],
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

def print_dataset_summary(
    dataset_name: str,
    rows: list[dict[str, Any]],
    output_file: Path,
) -> None:
    class_counts = Counter(
        row["label"]
        for row in rows
    )

    print(f"\n{dataset_name}")
    print("-" * len(dataset_name))
    print("Output file:", output_file)
    print("Total rows:", len(rows))
    print("Class counts:", dict(class_counts))

    if (
        rows
        and "anomaly_ground_truth" in rows[0]
    ):
        anomaly_count = sum(
            row["anomaly_ground_truth"]
            for row in rows
        )

        anomaly_rate = (
            anomaly_count / len(rows)
        )

        print(
            "Anomaly ground-truth rate:",
            round(anomaly_rate, 4),
        )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    training_rows = generate_dataset(
        split="train",
        class_counts=TRAIN_CLASS_COUNTS,
        random_seed=TRAIN_RANDOM_SEED,
    )

    testing_rows = generate_dataset(
        split="test",
        class_counts=TEST_CLASS_COUNTS,
        random_seed=TEST_RANDOM_SEED,
    )

    validate_dataset(
        rows=training_rows,
        expected_class_counts=TRAIN_CLASS_COUNTS,
        expected_fieldnames=TRAIN_FIELDNAMES,
        dataset_name="Training dataset",
    )

    validate_dataset(
        rows=testing_rows,
        expected_class_counts=TEST_CLASS_COUNTS,
        expected_fieldnames=TEST_FIELDNAMES,
        dataset_name="Final holdout dataset",
    )

    write_csv(
        rows=training_rows,
        output_file=TRAIN_FILE,
        fieldnames=TRAIN_FIELDNAMES,
    )

    write_csv(
        rows=testing_rows,
        output_file=TEST_FILE,
        fieldnames=TEST_FIELDNAMES,
    )

    print_dataset_summary(
        dataset_name=(
            "Balanced development/training dataset"
        ),
        rows=training_rows,
        output_file=TRAIN_FILE,
    )

    print_dataset_summary(
        dataset_name=(
            "Realistically imbalanced final "
            "holdout dataset"
        ),
        rows=testing_rows,
        output_file=TEST_FILE,
    )

    print("\nImportant:")
    print(
        "- timestamp is metadata and is not "
        "a model-input feature."
    )
    print(
        "- label is the supervised model's "
        "ground-truth target."
    )
    print(
        "- anomaly_ground_truth exists only "
        "in the final holdout file."
    )
    print(
        "- anomaly_ground_truth must never "
        "be provided to either model as input."
    )


if __name__ == "__main__":
    main()