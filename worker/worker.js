// Azure API that runs Gradient Boosting and Isolation Forest.
const AZURE_ML_API_URL =
  "https://salience-bot-ml-api-dyhkdcapcbg2grc9.canadacentral-01.azurewebsites.net/predict";


export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders() });
    }

    const url = new URL(request.url);

    // Return the latest telemetry events for the dashboard.
    if (request.method === "GET" && url.pathname === "/events") {
      try {
        const result = await env.DB.prepare(`
          SELECT *
          FROM telemetry_events
          ORDER BY id DESC
          LIMIT 100
        `).all();

        return jsonResponse({ events: result.results || [] });
      } catch (error) {
        return jsonResponse(
          {
            error: "Could not read events from the D1 database.",
            details: String(error),
          },
          500
        );
      }
    }

    // Basic Worker status endpoint.
    if (request.method === "GET" && url.pathname === "/") {
      return jsonResponse({
        status: "running",
        service: "Salience Telemetry Worker",
        message: "Use POST / for telemetry or GET /events for dashboard data.",
        models: {
          supervised: "Gradient Boosting",
          unsupervised: "Isolation Forest",
        },
      });
    }

    if (request.method !== "POST") {
      return jsonResponse({ error: "Method not allowed." }, 405);
    }

    let data;

    try {
      data = await request.json();
    } catch (error) {
      return jsonResponse(
        {
          error: "Invalid JSON body.",
          details: String(error),
        },
        400
      );
    }

    if (!data || typeof data !== "object" || Array.isArray(data)) {
      return jsonResponse(
        {
          error: "The JSON body must be an object.",
        },
        400
      );
    }

    // Only the minimized 14-feature schema is sent to the ML API.
    const telemetry = {
      page_category: data.page_category || "unknown_page",
      interaction_type: data.interaction_type || "page_view",
      scroll_depth_category: data.scroll_depth_category || "medium",
      request_interval_seconds: numberOrDefault(
        data.request_interval_seconds,
        10
      ),
      user_agent_category: data.user_agent_category || "unknown",
      has_favicon_request: numberOrDefault(data.has_favicon_request, 1),
      requested_robots_txt: numberOrDefault(data.requested_robots_txt, 0),
      pages_per_session: numberOrDefault(data.pages_per_session, 3),
      error_rate: numberOrDefault(data.error_rate, 0),
      tls_version: data.tls_version || "TLS1.3",
      cipher_suite_count: numberOrDefault(data.cipher_suite_count, 15),
      extension_count: numberOrDefault(data.extension_count, 12),
      alpn: data.alpn || "h2",
      sni_present: numberOrDefault(data.sni_present, 1),
    };

    let azureResult;

    try {
      const azureResponse = await fetch(AZURE_ML_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(telemetry),
      });

      azureResult = await readJsonResponse(azureResponse);

      if (!azureResponse.ok) {
        throw new Error(
          azureResult.error ||
            azureResult.details ||
            `Azure API returned HTTP ${azureResponse.status}`
        );
      }
    } catch (error) {
      azureResult = {
        ml_prediction: "azure_api_error",
        confidence: 0,
        class_probabilities: {},
        isolation_prediction: "unknown",
        anomaly_detected: false,
        isolation_decision_score: 0,
        isolation_threshold: 0,
        error: String(error),
      };
    }

    const workerPrediction =
      azureResult.ml_prediction ||
      azureResult.prediction ||
      "unknown";

    const confidence = numberOrDefault(
      azureResult.confidence ?? azureResult.probability,
      0
    );

    const classProbabilities =
      azureResult.class_probabilities &&
      typeof azureResult.class_probabilities === "object"
        ? azureResult.class_probabilities
        : {};

    const isolationPrediction =
      azureResult.isolation_prediction || "unknown";

    const anomalyDetected = toBoolean(
      azureResult.anomaly_detected
    );

    const isolationDecisionScore = numberOrDefault(
      azureResult.isolation_decision_score,
      0
    );

    const isolationThreshold = numberOrDefault(
      azureResult.isolation_threshold,
      0
    );

    const riskAnalysis = calculateRisk(
      workerPrediction,
      confidence,
      anomalyDetected
    );

    const timestamp = new Date().toISOString();

    try {
      await env.DB.prepare(`
        INSERT INTO telemetry_events (
          timestamp,
          page_category,
          interaction_type,
          scroll_depth_category,
          request_interval_seconds,
          user_agent_category,
          has_favicon_request,
          requested_robots_txt,
          pages_per_session,
          error_rate,
          tls_version,
          cipher_suite_count,
          extension_count,
          alpn,
          sni_present,
          worker_prediction,
          isolation_prediction,
          anomaly_detected,
          isolation_decision_score,
          risk_score,
          risk_level,
          action
        )
        VALUES (
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
          ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
      `)
        .bind(
          timestamp,
          telemetry.page_category,
          telemetry.interaction_type,
          telemetry.scroll_depth_category,
          telemetry.request_interval_seconds,
          telemetry.user_agent_category,
          telemetry.has_favicon_request,
          telemetry.requested_robots_txt,
          telemetry.pages_per_session,
          telemetry.error_rate,
          telemetry.tls_version,
          telemetry.cipher_suite_count,
          telemetry.extension_count,
          telemetry.alpn,
          telemetry.sni_present,
          workerPrediction,
          isolationPrediction,
          anomalyDetected ? 1 : 0,
          isolationDecisionScore,
          riskAnalysis.risk_score,
          riskAnalysis.risk_level,
          riskAnalysis.action
        )
        .run();
    } catch (error) {
      return jsonResponse(
        {
          error: "Could not save the event to the D1 database.",
          details: String(error),
          supervised_prediction: workerPrediction,
          supervised_confidence: confidence,
          isolation_prediction: isolationPrediction,
          anomaly_detected: anomalyDetected,
          isolation_decision_score: isolationDecisionScore,
          isolation_threshold: isolationThreshold,
          received_minimized_telemetry: telemetry,
        },
        500
      );
    }

    return jsonResponse({
      supervised: {
        model: "Gradient Boosting supervised classifier",
        prediction: workerPrediction,
        confidence,
        class_probabilities: classProbabilities,
      },

      unsupervised: {
        model: "Isolation Forest anomaly detector",
        prediction: isolationPrediction,
        anomaly_detected: anomalyDetected,
        decision_score: isolationDecisionScore,
        threshold: isolationThreshold,
      },

      risk_analysis: riskAnalysis,
      saved_to_database: true,
      received_minimized_telemetry: telemetry,

      privacy_note:
        "Only minimized telemetry is processed. No passwords, cookies, " +
        "tokens, names, emails, private content, exact location, or raw " +
        "IP addresses are collected.",

      timestamp,
    });
  },
};


// Calculate the final risk level using both model results.
function calculateRisk(prediction, confidence, anomalyDetected) {
  const confidencePercent =
    confidence <= 1
      ? Math.round(confidence * 100)
      : Math.round(confidence);

  let riskScore;
  let riskLevel;
  let action;

  if (prediction === "bad_bot" || prediction === "scanner") {
    riskScore = Math.max(70, confidencePercent);
    riskLevel = "high";
    action = "flag_for_review";
  } else if (prediction === "good_bot") {
    riskScore = 10;
    riskLevel = "low";
    action = "allow_with_monitoring";
  } else if (prediction === "human") {
    riskScore = 0;
    riskLevel = "low";
    action = "allow";
  } else {
    riskScore = 50;
    riskLevel = "medium";
    action = "monitor";
  }

  // An anomaly increases risk but does not replace the supervised class.
  if (anomalyDetected && prediction !== "azure_api_error") {
    riskScore = Math.max(riskScore, 50);

    if (riskLevel === "low") {
      riskLevel = "medium";
    }

    if (action === "allow" || action === "allow_with_monitoring") {
      action = "monitor_anomaly";
    }
  }

  return {
    risk_score: Math.max(0, Math.min(100, riskScore)),
    risk_level: riskLevel,
    action,
  };
}


// Preserve valid zero values while applying defaults to invalid numbers.
function numberOrDefault(value, defaultValue) {
  if (value === null || value === undefined || value === "") {
    return defaultValue;
  }

  const number = Number(value);
  return Number.isFinite(number) ? number : defaultValue;
}


function toBoolean(value) {
  return value === true || value === 1 || value === "true";
}


async function readJsonResponse(response) {
  const responseText = await response.text();

  if (!responseText) {
    return {};
  }

  try {
    return JSON.parse(responseText);
  } catch {
    throw new Error(
      `Azure API returned a non-JSON response with HTTP ${response.status}.`
    );
  }
}


function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders(),
    },
  });
}


function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}