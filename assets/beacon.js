// Cloudflare Worker endpoint that receives telemetry events.
const WORKER_URL =
    "https://salience-beacon-worker.mahin0710.workers.dev";

/**
 * Displays the current request or response inside the HTML element
 * with id="result".
 */
function showResult(payload) {
    const resultBox = document.getElementById("result");

    if (resultBox) {
        resultBox.textContent = JSON.stringify(payload, null, 2);
    }
}

/**
 * Sends a minimized telemetry event to the Cloudflare Worker.
 */
async function sendBeaconEvent(telemetry) {
    const safeTelemetry = {
        ...telemetry,

        // Use the current page when no page path was supplied.
        page_path:
            telemetry.page_path || window.location.pathname,

        // Optional client-side timestamp.
        // The Worker also creates its own timestamp before saving to D1.
        timestamp: new Date().toISOString()
    };

    showResult({
        status: "sending",
        message:
            "Sending a controlled, minimized telemetry event to the Cloudflare Worker.",
        event: safeTelemetry
    });

    try {
        const response = await fetch(WORKER_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(safeTelemetry),
            cache: "no-store"
        });

        const responseText = await response.text();

        let workerResult;

        try {
            workerResult = JSON.parse(responseText);
        } catch {
            workerResult = {
                response_text: responseText
            };
        }

        showResult({
            status: response.ok ? "success" : "worker_error",
            worker_status: response.status,
            worker_endpoint: WORKER_URL,
            result: workerResult
        });
    } catch (error) {
        showResult({
            status: "network_error",
            message:
                "Could not send the telemetry event to the Cloudflare Worker.",
            error: String(error)
        });
    }
}

/**
 * Simulated normal human browsing event.
 */
function humanTelemetry() {
    return {
        page_path: window.location.pathname,
        interaction_type: "normal_browsing",
        scroll_depth_category: "medium",
        request_interval_seconds: 5,
        user_agent_category: "normal_browser",
        has_favicon_request: 1,
        requested_robots_txt: 0,
        pages_per_session: 4,
        error_rate: 0.0,
        tls_version: "TLS1.3",
        cipher_suite_count: 15,
        extension_count: 12,
        alpn: "h2",
        sni_present: 1
    };
}

/**
 * Simulated legitimate crawler / good bot event.
 */
function goodBotTelemetry() {
    return {
        page_path: "/robots.txt",
        interaction_type: "crawler_request",
        scroll_depth_category: "none",
        request_interval_seconds: 2,
        user_agent_category: "good_bot",
        has_favicon_request: 0,
        requested_robots_txt: 1,
        pages_per_session: 15,
        error_rate: 0.01,
        tls_version: "TLS1.3",
        cipher_suite_count: 12,
        extension_count: 10,
        alpn: "h2",
        sni_present: 1
    };
}

/**
 * Simulated malicious bot event.
 */
function badBotTelemetry() {
    return {
        page_path: "/products.html",
        interaction_type: "rapid_request",
        scroll_depth_category: "none",
        request_interval_seconds: 0.2,
        user_agent_category: "bot",
        has_favicon_request: 0,
        requested_robots_txt: 0,
        pages_per_session: 100,
        error_rate: 0.2,
        tls_version: "TLS1.2",
        cipher_suite_count: 5,
        extension_count: 4,
        alpn: "http/1.1",
        sni_present: 1
    };
}

/**
 * Simulated vulnerability scanner event.
 */
function scannerTelemetry() {
    return {
        page_path: "/admin",
        interaction_type: "scanner_request",
        scroll_depth_category: "none",
        request_interval_seconds: 0.1,
        user_agent_category: "curl",
        has_favicon_request: 0,
        requested_robots_txt: 0,
        pages_per_session: 150,
        error_rate: 0.6,
        tls_version: "TLS1.2",
        cipher_suite_count: 4,
        extension_count: 3,
        alpn: "http/1.1",
        sni_present: 0
    };
}

/**
 * Connect each HTML button to its corresponding telemetry event.
 */
function initializeTelemetryButtons() {
    const buttonMappings = [
        ["humanButton", humanTelemetry],
        ["goodBotButton", goodBotTelemetry],
        ["badBotButton", badBotTelemetry],
        ["scannerButton", scannerTelemetry]
    ];

    buttonMappings.forEach(([buttonId, telemetryFactory]) => {
        const button = document.getElementById(buttonId);

        if (button) {
            button.addEventListener("click", () => {
                sendBeaconEvent(telemetryFactory());
            });
        }
    });
}

// Wait until the HTML page has loaded before searching for buttons.
document.addEventListener(
    "DOMContentLoaded",
    initializeTelemetryButtons
);