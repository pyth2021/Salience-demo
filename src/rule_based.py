"""Rule-based baseline detector for Salience bot/anomaly detection.

This module intentionally uses minimized telemetry features only.
It does not require labels for prediction.

Scope note:
The current capstone prototype focuses on classification, risk scoring,
monitoring, reporting, and audit evidence. It does not perform production-grade
blocking, CAPTCHA enforcement, rate limiting, or active mitigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RuleResult:
    worker_prediction: str
    risk_score: int
    risk_level: str
    action: str
    matched_rules: List[str]


def score_event(event: Dict[str, Any]) -> RuleResult:
    """Score one telemetry event using rule-based detection logic."""

    risk_score = 0
    matched_rules: List[str] = []

    interaction_type = str(event.get("interaction_type", ""))
    user_agent_category = str(event.get("user_agent_category", "normal_browser"))

    request_interval_seconds = float(event.get("request_interval_seconds", 10))
    pages_per_session = int(event.get("pages_per_session", 1))
    error_rate = float(event.get("error_rate", 0))

    has_favicon_request = int(event.get("has_favicon_request", 1))
    requested_robots_txt = int(event.get("requested_robots_txt", 0))

    cipher_suite_count = int(event.get("cipher_suite_count", 12))
    extension_count = int(event.get("extension_count", 10))

    # Scanner-like user agents or request patterns
    if user_agent_category in {"curl", "python", "script"} or interaction_type == "scanner_request":
        risk_score += 40
        matched_rules.append("scanner_path_or_user_agent")

    # Very fast request timing
    if request_interval_seconds < 1.0:
        risk_score += 25
        matched_rules.append("rapid_request_interval")

    # Abnormally high session activity
    if pages_per_session >= 60:
        risk_score += 20
        matched_rules.append("high_pages_per_session")

    # High error rate can suggest probing/scanning
    if error_rate >= 0.20:
        risk_score += 20
        matched_rules.append("high_error_rate")

    # Missing normal browser behaviour signals
    if has_favicon_request == 0 and requested_robots_txt == 0:
        risk_score += 10
        matched_rules.append("no_favicon_or_robots")

    # Simplified TLS-inspired anomaly signal
    if cipher_suite_count <= 6 or extension_count <= 5:
        risk_score += 10
        matched_rules.append("weak_tls_profile")

    risk_score = min(risk_score, 100)

    if risk_score >= 60:
        risk_level = "high"
        action = "flag_for_review"
        worker_prediction = "bad_bot_or_scanner"
    elif risk_score >= 30:
        risk_level = "medium"
        action = "monitor"
        worker_prediction = "suspicious"
    else:
        risk_level = "low"
        action = "allow"
        worker_prediction = "human_or_good_bot"

    return RuleResult(
        worker_prediction=worker_prediction,
        risk_score=risk_score,
        risk_level=risk_level,
        action=action,
        matched_rules=matched_rules,
    )


if __name__ == "__main__":
    sample_event = {
        "interaction_type": "scanner_request",
        "user_agent_category": "curl",
        "request_interval_seconds": 0.3,
        "pages_per_session": 120,
        "error_rate": 0.45,
        "has_favicon_request": 0,
        "requested_robots_txt": 0,
        "cipher_suite_count": 4,
        "extension_count": 3,
    }

    result = score_event(sample_event)

    print("Rule-based detector test result:")
    print(f"Prediction: {result.worker_prediction}")
    print(f"Risk score: {result.risk_score}")
    print(f"Risk level: {result.risk_level}")
    print(f"Action: {result.action}")
    print(f"Matched rules: {result.matched_rules}")