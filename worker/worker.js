// Azure ML API endpoint.
// This is the cloud ML model deployed on Azure App Service.
const AZURE_ML_API_URL =
  "https://salience-bot-ml-api-dyhkdcapcbg2grc9.canadacentral-01.azurewebsites.net/predict";

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight requests.
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: corsHeaders(),
      });
    }

    const url = new URL(request.url);

    // GET /events is used by the Streamlit dashboard to read live D1 data.
    if (request.method === "GET" && url.pathname === "/events") {
      try {
        const result = await env.DB.prepare(
          "SELECT * FROM telemetry_events ORDER BY id DESC LIMIT 100"
        ).all();

        return jsonResponse({
          events: result.results || [],
        });
      } catch (error) {
        return jsonResponse(
          {
            error: "Could not read events from D1 database.",
            details: String(error),
          },
          500
        );
      }
    }

    // GET / is only a simple status message.
    if (request.method === "GET" && url.pathname === "/") {
      return jsonResponse({
        status: "running",
        message:
          "Cloudflare Worker is running. Use POST / for beacon events or GET /events for dashboard data.",
      });
    }

    // Only POST requests are allowed for telemetry events.
    if (request.method !== "POST") {
      return jsonResponse(
        {
          error: "Method not allowed",
        },
        405
      );
    }

    // Read telemetry data sent from beacon.js, sandbox, or dashboard test.
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

    // Privacy validation / minimized telemetry schema.
    // Only approved non-sensitive fields are forwarded and stored.
    const telemetry = {
      page_path: data.page_path || "/",
      interaction_type: data.interaction_type || "unknown",
      scroll_depth_category: data.scroll_depth_category || "unknown",
      request_interval_seconds: Number(data.request_interval_seconds || 0),
      user_agent_category: data.user_agent_category || "unknown",
      has_favicon_request: Number(data.has_favicon_request || 0),
      requested_robots_txt: Number(data.requested_robots_txt || 0),
      pages_per_session: Number(data.pages_per_session || 0),
      error_rate: Number(data.error_rate || 0),
      tls_version: data.tls_version || "unknown",
      cipher_suite_count: Number(data.cipher_suite_count || 0),
      extension_count: Number(data.extension_count || 0),
      alpn: data.alpn || "unknown",
      sni_present: Number(data.sni_present || 0),
    };

    // Send minimized telemetry to Azure ML API.
    let azureResult = {};

    try {
      const azureResponse = await fetch(AZURE_ML_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(telemetry),
      });

      azureResult = await azureResponse.json();
    } catch (error) {
      azureResult = {
        ml_prediction: "azure_api_error",
        confidence: 0,
        error: String(error),
      };
    }

    // Support different possible response field names from Azure.
    const workerPrediction =
      azureResult.ml_prediction || azureResult.prediction || "unknown";

    const confidence = Number(
      azureResult.confidence || azureResult.probability || 0
    );

    let riskScore = 0;
    let riskLevel = "low";
    let action = "allow";

    // High-risk traffic.
    // Supports both "scanner" and "scanner_like" for consistency.
    if (
      workerPrediction === "bad_bot" ||
      workerPrediction === "scanner" ||
      workerPrediction === "scanner_like" ||
      workerPrediction === "bad_bot_or_scanner"
    ) {
      riskScore =
        confidence <= 1
          ? Math.round(confidence * 100)
          : Math.round(confidence);

      riskLevel = "high";
      action = "flag_for_review";
    }

    // Good bot traffic.
    else if (workerPrediction === "good_bot") {
      riskScore = 10;
      riskLevel = "low";
      action = "allow_with_monitoring";
    }

    // Human or benign traffic.
    else if (
      workerPrediction === "human" ||
      workerPrediction === "human_or_benign" ||
      workerPrediction === "human_or_good_bot"
    ) {
      riskScore = 0;
      riskLevel = "low";
      action = "allow";
    }

    // Azure/API error or unknown result.
    else {
      riskScore = 50;
      riskLevel = "medium";
      action = "monitor";
    }

    // Save the minimized event summary into Cloudflare D1.
    // No passwords, cookies, tokens, names, emails, private content,
    // exact location, or raw IP addresses are stored.
    try {
      await env.DB.prepare(
        `INSERT INTO telemetry_events (
          timestamp,
          page_path,
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
          risk_score,
          risk_level,
          action
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
        .bind(
          new Date().toISOString(),
          telemetry.page_path,
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
          riskScore,
          riskLevel,
          action
        )
        .run();
    } catch (error) {
      return jsonResponse(
        {
          error: "Could not save event to D1 database.",
          details: String(error),
          azure_ml_prediction: workerPrediction,
          azure_confidence: confidence,
          received_minimized_telemetry: telemetry,
        },
        500
      );
    }

    return jsonResponse({
      azure_ml_prediction: workerPrediction,
      azure_confidence: confidence,
      risk_score: riskScore,
      risk_level: riskLevel,
      action: action,
      saved_to_database: true,
      received_minimized_telemetry: telemetry,
      privacy_note:
        "Only minimized telemetry is processed. No passwords, cookies, tokens, names, emails, private content, exact location, or raw IP addresses are collected.",
      timestamp: new Date().toISOString(),
    });
  },
};

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data, null, 2), {
    status: status,
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