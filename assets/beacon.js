const WORKER_URL = "https://salience-beacon-worker.mahin0710.workers.dev";

function showResult(payload) {
    const resultBox = document.getElementById("result");
    if (resultBox) {
        resultBox.innerText = JSON.stringify(payload, null, 2);
    }
}

async function sendBeaconEvent(telemetry) {
    const safeTelemetry = {
        ...telemetry,
        page_path: telemetry.page_path || window.location.pathname,
        event_source: "salience-demo-pages",
        client_timestamp: new Date().toISOString()
    };

    showResult({
        status: "sending",
        message: "Sending controlled minimized telemetry event to the Cloudflare Worker.",
        event: safeTelemetry
    });

    try {
        const response = await fetch(WORKER_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(safeTelemetry)
        });

        const text = await response.text();
        let result;
        try {
            result = JSON.parse(text);
        } catch (error) {
            result = { status: response.status, response_text: text };
        }

        showResult({
            worker_status: response.status,
            worker_endpoint: WORKER_URL,
            result
        });
    } catch (error) {
        showResult({
            status: "error",
            message: "Could not send telemetry event to the Cloudflare Worker.",
            error: String(error)
        });
    }
}

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
        error_rate: 0.20,
        tls_version: "TLS1.2",
        cipher_suite_count: 5,
        extension_count: 4,
        alpn: "http/1.1",
        sni_present: 1
    };
}

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
        error_rate: 0.60,
        tls_version: "TLS1.2",
        cipher_suite_count: 4,
        extension_count: 3,
        alpn: "http/1.1",
        sni_present: 0
    };
}

const buttonMappings = [
    ["humanButton", humanTelemetry],
    ["goodBotButton", goodBotTelemetry],
    ["badBotButton", badBotTelemetry],
    ["scannerButton", scannerTelemetry]
];

buttonMappings.forEach(([id, telemetryFactory]) => {
    const button = document.getElementById(id);
    if (button) {
        button.addEventListener("click", () => sendBeaconEvent(telemetryFactory()));
    }
});
