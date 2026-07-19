// Cloudflare Worker endpoint that receives telemetry events.
const WORKER_URL = "https://salience-beacon-worker.mahin0710.workers.dev";


// -----------------------------------------------------------------------------
// RANDOMIZATION HELPERS
// -----------------------------------------------------------------------------

function randomFloat(minimum, maximum, decimalPlaces = 2) {
    const value = minimum + Math.random() * (maximum - minimum);
    return Number(value.toFixed(decimalPlaces));
}

function randomInteger(minimum, maximum) {
    return Math.floor(Math.random() * (maximum - minimum + 1)) + minimum;
}

function probabilityFlag(probability) {
    return Math.random() < probability ? 1 : 0;
}

function weightedChoice(options) {
    const totalWeight = options.reduce((total, [, weight]) => total + weight, 0);
    let randomValue = Math.random() * totalWeight;

    for (const [value, weight] of options) {
        randomValue -= weight;
        if (randomValue <= 0) return value;
    }

    return options[options.length - 1][0];
}

function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
}

function chooseValue(value) {
    return Array.isArray(value) ? weightedChoice(value) : value;
}


// -----------------------------------------------------------------------------
// NETWORK PROFILE
// -----------------------------------------------------------------------------

function createNetworkProfile(userAgentCategory, settings = {}) {
    const tls13Probability = settings.tls13Probability ?? 0.8;
    const baseH2Probability = settings.h2Probability ?? 0.7;
    const sniProbability = settings.sniProbability ?? 0.98;

    const tlsVersion = Math.random() < tls13Probability ? "TLS1.3" : "TLS1.2";

    const technicalRanges = {
        browser: [10, 18, 8, 16],
        crawler: [8, 16, 7, 14],
        script_client: [4, 12, 3, 10],
        command_line: [2, 9, 2, 8],
        unknown: [4, 14, 3, 12]
    };

    const ranges = technicalRanges[userAgentCategory] || technicalRanges.unknown;

    let cipherSuiteCount = randomInteger(ranges[0], ranges[1]);
    let extensionCount = randomInteger(ranges[2], ranges[3]);

    if (tlsVersion === "TLS1.3") {
        cipherSuiteCount += probabilityFlag(0.35);
        extensionCount += probabilityFlag(0.40);
    }

    let h2Probability = baseH2Probability;
    h2Probability += tlsVersion === "TLS1.3" ? 0.05 : -0.05;

    if (userAgentCategory === "browser") h2Probability += 0.05;

    if (
        userAgentCategory === "script_client" ||
        userAgentCategory === "command_line"
    ) {
        h2Probability -= 0.15;
    }

    if (extensionCount <= 4) h2Probability -= 0.15;
    if (extensionCount >= 11) h2Probability += 0.04;

    h2Probability = clamp(h2Probability, 0.02, 0.97);

    return {
        tls_version: tlsVersion,
        cipher_suite_count: cipherSuiteCount,
        extension_count: extensionCount,
        alpn: Math.random() < h2Probability ? "h2" : "http/1.1",
        sni_present: probabilityFlag(sniProbability)
    };
}


// -----------------------------------------------------------------------------
// TELEMETRY PROFILE GENERATOR
// -----------------------------------------------------------------------------

function createTelemetry(profile) {
    const userAgentCategory = chooseValue(profile.userAgent);

    return {
        page_category: chooseValue(profile.page),
        interaction_type: chooseValue(profile.interaction),
        scroll_depth_category: chooseValue(profile.scroll),
        request_interval_seconds: randomFloat(...profile.interval),
        user_agent_category: userAgentCategory,
        has_favicon_request: probabilityFlag(profile.favicon),
        requested_robots_txt: probabilityFlag(profile.robots),
        pages_per_session: randomInteger(...profile.pages),
        error_rate: randomFloat(...profile.error),
        ...createNetworkProfile(userAgentCategory, profile.network)
    };
}


// -----------------------------------------------------------------------------
// HUMAN PROFILES
// -----------------------------------------------------------------------------

const HUMAN_PROFILES = {
    normal_reader: {
        page: [
            ["public_page", 0.72],
            ["account_page", 0.15],
            ["checkout_page", 0.08],
            ["unknown_page", 0.05]
        ],
        interaction: [
            ["navigation", 0.45],
            ["page_view", 0.35],
            ["form_request", 0.10],
            ["resource_request", 0.10]
        ],
        scroll: [
            ["low", 0.20],
            ["medium", 0.50],
            ["high", 0.25],
            ["none", 0.05]
        ],
        interval: [4, 10],
        userAgent: "browser",
        favicon: 0.92,
        robots: 0.01,
        pages: [3, 10],
        error: [0, 0.05, 3],
        network: {
            tls13Probability: 0.90,
            h2Probability: 0.82,
            sniProbability: 0.995
        }
    },

    fast_navigator: {
        page: [
            ["public_page", 0.60],
            ["account_page", 0.20],
            ["checkout_page", 0.12],
            ["unknown_page", 0.08]
        ],
        interaction: [
            ["navigation", 0.55],
            ["page_view", 0.25],
            ["api_request", 0.10],
            ["resource_request", 0.10]
        ],
        scroll: [
            ["low", 0.40],
            ["medium", 0.40],
            ["high", 0.15],
            ["none", 0.05]
        ],
        interval: [1.5, 4],
        userAgent: "browser",
        favicon: 0.85,
        robots: 0.01,
        pages: [8, 18],
        error: [0.01, 0.08, 3],
        network: {
            tls13Probability: 0.88,
            h2Probability: 0.78,
            sniProbability: 0.99
        }
    },

    privacy_focused: {
        page: [
            ["public_page", 0.60],
            ["account_page", 0.20],
            ["unknown_page", 0.20]
        ],
        interaction: [
            ["navigation", 0.45],
            ["page_view", 0.35],
            ["api_request", 0.20]
        ],
        scroll: [
            ["low", 0.35],
            ["medium", 0.45],
            ["high", 0.15],
            ["none", 0.05]
        ],
        interval: [3, 8],
        userAgent: [
            ["browser", 0.65],
            ["unknown", 0.35]
        ],
        favicon: 0.35,
        robots: 0.01,
        pages: [2, 9],
        error: [0.01, 0.09, 3],
        network: {
            tls13Probability: 0.80,
            h2Probability: 0.55,
            sniProbability: 0.98
        }
    }
};


// -----------------------------------------------------------------------------
// GOOD-BOT PROFILES
// -----------------------------------------------------------------------------

const GOOD_BOT_PROFILES = {
    search_crawler: {
        page: [
            ["public_page", 0.50],
            ["crawler_file", 0.45],
            ["unknown_page", 0.05]
        ],
        interaction: [
            ["automated_request", 0.45],
            ["resource_request", 0.35],
            ["api_request", 0.20]
        ],
        scroll: "none",
        interval: [1.5, 5],
        userAgent: [
            ["crawler", 0.85],
            ["unknown", 0.15]
        ],
        favicon: 0.05,
        robots: 0.90,
        pages: [15, 60],
        error: [0, 0.08, 3],
        network: {
            tls13Probability: 0.85,
            h2Probability: 0.75,
            sniProbability: 0.99
        }
    },

    sitemap_crawler: {
        page: [
            ["crawler_file", 0.75],
            ["public_page", 0.20],
            ["unknown_page", 0.05]
        ],
        interaction: [
            ["automated_request", 0.60],
            ["resource_request", 0.40]
        ],
        scroll: "none",
        interval: [2, 6],
        userAgent: "crawler",
        favicon: 0.03,
        robots: 0.98,
        pages: [10, 45],
        error: [0, 0.06, 3],
        network: {
            tls13Probability: 0.86,
            h2Probability: 0.74,
            sniProbability: 0.995
        }
    },

    monitoring_bot: {
        page: [
            ["public_page", 0.60],
            ["account_page", 0.15],
            ["checkout_page", 0.10],
            ["unknown_page", 0.15]
        ],
        interaction: [
            ["page_view", 0.30],
            ["api_request", 0.35],
            ["resource_request", 0.25],
            ["automated_request", 0.10]
        ],
        scroll: [
            ["none", 0.80],
            ["low", 0.20]
        ],
        interval: [4, 12],
        userAgent: [
            ["browser", 0.35],
            ["crawler", 0.25],
            ["script_client", 0.30],
            ["unknown", 0.10]
        ],
        favicon: 0.40,
        robots: 0.30,
        pages: [3, 15],
        error: [0, 0.12, 3],
        network: {
            tls13Probability: 0.85,
            h2Probability: 0.70,
            sniProbability: 0.99
        }
    }
};


// -----------------------------------------------------------------------------
// BAD-BOT PROFILES
// -----------------------------------------------------------------------------

const BAD_BOT_PROFILES = {
    aggressive_scraper: {
        page: [
            ["public_page", 0.55],
            ["account_page", 0.15],
            ["sensitive_page", 0.10],
            ["unknown_page", 0.20]
        ],
        interaction: [
            ["automated_request", 0.50],
            ["resource_request", 0.25],
            ["api_request", 0.25]
        ],
        scroll: "none",
        interval: [0.08, 0.60],
        userAgent: [
            ["script_client", 0.50],
            ["unknown", 0.30],
            ["command_line", 0.20]
        ],
        favicon: 0.10,
        robots: 0.03,
        pages: [70, 160],
        error: [0.12, 0.35, 3],
        network: {
            tls13Probability: 0.65,
            h2Probability: 0.35,
            sniProbability: 0.95
        }
    },

    stealth_scraper: {
        page: [
            ["public_page", 0.65],
            ["account_page", 0.15],
            ["checkout_page", 0.10],
            ["unknown_page", 0.10]
        ],
        interaction: [
            ["navigation", 0.25],
            ["page_view", 0.20],
            ["resource_request", 0.25],
            ["automated_request", 0.30]
        ],
        scroll: [
            ["none", 0.50],
            ["low", 0.35],
            ["medium", 0.15]
        ],
        interval: [1.5, 5],
        userAgent: [
            ["browser", 0.60],
            ["unknown", 0.20],
            ["script_client", 0.20]
        ],
        favicon: 0.65,
        robots: 0.08,
        pages: [20, 70],
        error: [0.03, 0.18, 3],
        network: {
            tls13Probability: 0.85,
            h2Probability: 0.75,
            sniProbability: 0.98
        }
    },

    credential_automation: {
        page: [
            ["account_page", 0.60],
            ["checkout_page", 0.20],
            ["sensitive_page", 0.10],
            ["unknown_page", 0.10]
        ],
        interaction: [
            ["form_request", 0.45],
            ["api_request", 0.30],
            ["automated_request", 0.25]
        ],
        scroll: "none",
        interval: [0.20, 1.20],
        userAgent: [
            ["script_client", 0.45],
            ["unknown", 0.30],
            ["browser", 0.25]
        ],
        favicon: 0.20,
        robots: 0.02,
        pages: [30, 100],
        error: [0.15, 0.40, 3],
        network: {
            tls13Probability: 0.72,
            h2Probability: 0.50,
            sniProbability: 0.96
        }
    }
};


// -----------------------------------------------------------------------------
// SCANNER PROFILES
// -----------------------------------------------------------------------------

const SCANNER_PROFILES = {
    fast_scanner: {
        page: [
            ["sensitive_page", 0.65],
            ["unknown_page", 0.25],
            ["account_page", 0.10]
        ],
        interaction: [
            ["automated_request", 0.60],
            ["api_request", 0.25],
            ["resource_request", 0.15]
        ],
        scroll: "none",
        interval: [0.03, 0.25],
        userAgent: [
            ["command_line", 0.40],
            ["script_client", 0.35],
            ["unknown", 0.25]
        ],
        favicon: 0.02,
        robots: 0.01,
        pages: [120, 240],
        error: [0.50, 0.90, 3],
        network: {
            tls13Probability: 0.55,
            h2Probability: 0.25,
            sniProbability: 0.90
        }
    },

    slow_scanner: {
        page: [
            ["sensitive_page", 0.55],
            ["unknown_page", 0.30],
            ["account_page", 0.15]
        ],
        interaction: [
            ["automated_request", 0.55],
            ["api_request", 0.30],
            ["resource_request", 0.15]
        ],
        scroll: "none",
        interval: [1.5, 5],
        userAgent: [
            ["unknown", 0.40],
            ["script_client", 0.35],
            ["command_line", 0.25]
        ],
        favicon: 0.05,
        robots: 0.01,
        pages: [50, 120],
        error: [0.25, 0.65, 3],
        network: {
            tls13Probability: 0.65,
            h2Probability: 0.35,
            sniProbability: 0.93
        }
    },

    evasive_scanner: {
        page: [
            ["sensitive_page", 0.45],
            ["unknown_page", 0.25],
            ["account_page", 0.20],
            ["public_page", 0.10]
        ],
        interaction: [
            ["automated_request", 0.45],
            ["api_request", 0.30],
            ["form_request", 0.25]
        ],
        scroll: [
            ["none", 0.75],
            ["low", 0.25]
        ],
        interval: [0.50, 2.50],
        userAgent: [
            ["browser", 0.25],
            ["unknown", 0.35],
            ["script_client", 0.25],
            ["command_line", 0.15]
        ],
        favicon: 0.25,
        robots: 0.03,
        pages: [60, 150],
        error: [0.20, 0.55, 3],
        network: {
            tls13Probability: 0.80,
            h2Probability: 0.60,
            sniProbability: 0.96
        }
    }
};


// -----------------------------------------------------------------------------
// EVENT FUNCTIONS
// -----------------------------------------------------------------------------

function humanTelemetry() {
    const profile = weightedChoice([
        ["normal_reader", 0.50],
        ["fast_navigator", 0.25],
        ["privacy_focused", 0.25]
    ]);

    return createTelemetry(HUMAN_PROFILES[profile]);
}

function goodBotTelemetry() {
    const profile = weightedChoice([
        ["search_crawler", 0.55],
        ["sitemap_crawler", 0.20],
        ["monitoring_bot", 0.25]
    ]);

    return createTelemetry(GOOD_BOT_PROFILES[profile]);
}

function badBotTelemetry() {
    const profile = weightedChoice([
        ["aggressive_scraper", 0.40],
        ["stealth_scraper", 0.35],
        ["credential_automation", 0.25]
    ]);

    return createTelemetry(BAD_BOT_PROFILES[profile]);
}

function scannerTelemetry() {
    const profile = weightedChoice([
        ["fast_scanner", 0.45],
        ["slow_scanner", 0.30],
        ["evasive_scanner", 0.25]
    ]);

    return createTelemetry(SCANNER_PROFILES[profile]);
}


// -----------------------------------------------------------------------------
// DISPLAY AND SEND TELEMETRY
// -----------------------------------------------------------------------------

function showResult(payload) {
    const resultBox = document.getElementById("result");

    if (resultBox) {
        resultBox.textContent = JSON.stringify(payload, null, 2);
    }
}

async function sendBeaconEvent(telemetry) {
    showResult({
        status: "sending",
        message: "Sending minimized telemetry to the Cloudflare Worker.",
        sent_at: new Date().toISOString(),
        event: telemetry
    });

    try {
        const response = await fetch(WORKER_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(telemetry),
            cache: "no-store"
        });

        const responseText = await response.text();
        let workerResult;

        try {
            workerResult = JSON.parse(responseText);
        } catch {
            workerResult = { response_text: responseText };
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
            message: "Could not send telemetry to the Cloudflare Worker.",
            error: String(error)
        });
    }
}


// -----------------------------------------------------------------------------
// CONNECT HTML BUTTONS
// -----------------------------------------------------------------------------

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

document.addEventListener("DOMContentLoaded", initializeTelemetryButtons);