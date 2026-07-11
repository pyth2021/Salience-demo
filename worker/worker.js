// Azure ML API endpoint.
// This API runs both Gradient Boosting and Isolation Forest.
const AZURE_ML_API_URL =
  "https://salience-bot-ml-api-dyhkdcapcbg2grc9.canadacentral-01.azurewebsites.net/predict";

export default {
  async fetch(request, env, ctx) {
    // ---------------------------------------------------------
    // CORS PREFLIGHT
    // ---------------------------------------------------------

    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: corsHeaders(),
      });
    }

    const url = new URL(request.url);


    // ---------------------------------------------------------
    // GET LIVE EVENTS FOR THE DASHBOARD
    // ---------------------------------------------------------

    if (
      request.method === "GET" &&
      url.pathname === "/events"
    ) {
      try {
        const result = await env.DB.prepare(
          `
          SELECT *
          FROM telemetry_events
          ORDER BY id DESC
          LIMIT 100
          `
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


    // ---------------------------------------------------------
    // WORKER STATUS ENDPOINT
    // ---------------------------------------------------------

    if (
      request.method === "GET" &&
      url.pathname === "/"
    ) {
      return jsonResponse({
        status: "running",
        service: "Salience Telemetry Worker",
        message:
          "Use POST / for telemetry events or GET /events for dashboard data.",
        models: {
          supervised: "Gradient Boosting",
          unsupervised: "Isolation Forest",
        },
      });
    }


    // ---------------------------------------------------------
    // ONLY ALLOW POST FOR TELEMETRY
    // ---------------------------------------------------------

    if (request.method !== "POST") {
      return jsonResponse(
        {
          error: "Method not allowed.",
        },
        405
      );
    }


    // ---------------------------------------------------------
    // READ REQUEST JSON
    // ---------------------------------------------------------

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


    // ---------------------------------------------------------
    // MINIMIZED TELEMETRY SCHEMA
    // ---------------------------------------------------------

    const telemetry = {
      page_path:
        data.page_path || "/",

      interaction_type:
        data.interaction_type || "unknown",

      scroll_depth_category:
        data.scroll_depth_category || "unknown",

      request_interval_seconds:
        Number(data.request_interval_seconds || 0),

      user_agent_category:
        data.user_agent_category || "unknown",

      has_favicon_request:
        Number(data.has_favicon_request || 0),

      requested_robots_txt:
        Number(data.requested_robots_txt || 0),

      pages_per_session:
        Number(data.pages_per_session || 0),

      error_rate:
        Number(data.error_rate || 0),

      tls_version:
        data.tls_version || "unknown",

      cipher_suite_count:
        Number(data.cipher_suite_count || 0),

      extension_count:
        Number(data.extension_count || 0),

      alpn:
        data.alpn || "unknown",

      sni_present:
        Number(data.sni_present || 0),
    };


    // ---------------------------------------------------------
    // CALL THE AZURE DUAL-MODEL API
    // ---------------------------------------------------------

    let azureResult = {};

    try {
      const azureResponse = await fetch(
        AZURE_ML_API_URL,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify(telemetry),
        }
      );

      azureResult = await azureResponse.json();

      if (!azureResponse.ok) {
        throw new Error(
          azureResult.error ||
          `Azure API returned HTTP ${azureResponse.status}`
        );
      }

    } catch (error) {
      azureResult = {
        ml_prediction: "azure_api_error",
        confidence: 0,

        isolation_prediction: "unknown",
        anomaly_detected: false,
        isolation_decision_score: 0,

        error: String(error),
      };
    }


    // ---------------------------------------------------------
    // READ GRADIENT BOOSTING RESULT
    // ---------------------------------------------------------

    const workerPrediction =
      azureResult.ml_prediction ||
      azureResult.prediction ||
      "unknown";

    const confidence = Number(
      azureResult.confidence ??
      azureResult.probability ??
      0
    );


    // ---------------------------------------------------------
    // READ ISOLATION FOREST RESULT
    // ---------------------------------------------------------

    const isolationPrediction =
      azureResult.isolation_prediction ||
      "unknown";

    const anomalyDetected =
      azureResult.anomaly_detected === true ||
      azureResult.anomaly_detected === 1 ||
      azureResult.anomaly_detected === "true";

    const isolationDecisionScore = Number(
      azureResult.isolation_decision_score ?? 0
    );


    // ---------------------------------------------------------
    // RULE-BASED RISK SCORE
    // ---------------------------------------------------------

    let riskScore = 0;
    let riskLevel = "low";
    let action = "allow";


    // High-risk supervised classifications.
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


    // Known good bot.
    else if (
      workerPrediction === "good_bot"
    ) {
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


    // Azure error or unknown prediction.
    else {
      riskScore = 50;
      riskLevel = "medium";
      action = "monitor";
    }


    // Isolation Forest adds anomaly context.
    // This does not replace the supervised classification.
    if (
      anomalyDetected &&
      workerPrediction !== "azure_api_error"
    ) {
      if (riskScore < 50) {
        riskScore = 50;
      }

      if (riskLevel === "low") {
        riskLevel = "medium";
      }

      if (action === "allow") {
        action = "monitor_anomaly";
      }
    }


    // Keep risk score inside the range 0–100.
    riskScore = Math.max(
      0,
      Math.min(100, riskScore)
    );


    // ---------------------------------------------------------
    // SAVE EVENT INTO CLOUDFLARE D1
    // ---------------------------------------------------------

    try {
      await env.DB.prepare(
        `
        INSERT INTO telemetry_events (
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
        `
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
          isolationPrediction,
          anomalyDetected ? 1 : 0,
          isolationDecisionScore,

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

          supervised_prediction:
            workerPrediction,

          supervised_confidence:
            confidence,

          isolation_prediction:
            isolationPrediction,

          anomaly_detected:
            anomalyDetected,

          isolation_decision_score:
            isolationDecisionScore,

          received_minimized_telemetry:
            telemetry,
        },
        500
      );
    }


    // ---------------------------------------------------------
    // RETURN WORKER RESPONSE
    // ---------------------------------------------------------

    return jsonResponse({
      supervised: {
        model:
          "Gradient Boosting supervised classifier",

        prediction:
          workerPrediction,

        confidence:
          confidence,

        class_probabilities:
          azureResult.class_probabilities || {},
      },

      unsupervised: {
        model:
          "Isolation Forest unsupervised anomaly detector",

        prediction:
          isolationPrediction,

        anomaly_detected:
          anomalyDetected,

        decision_score:
          isolationDecisionScore,
      },

      risk_analysis: {
        risk_score:
          riskScore,

        risk_level:
          riskLevel,

        action:
          action,
      },

      saved_to_database: true,

      received_minimized_telemetry:
        telemetry,

      privacy_note:
        "Only minimized telemetry is processed. No passwords, cookies, tokens, names, emails, private content, exact location, or raw IP addresses are collected.",

      timestamp:
        new Date().toISOString(),
    });
  },
};


// ---------------------------------------------------------
// JSON RESPONSE HELPER
// ---------------------------------------------------------

function jsonResponse(
  data,
  status = 200
) {
  return new Response(
    JSON.stringify(
      data,
      null,
      2
    ),
    {
      status: status,

      headers: {
        "Content-Type": "application/json",
        ...corsHeaders(),
      },
    }
  );
}


// ---------------------------------------------------------
// CORS HEADERS
// ---------------------------------------------------------

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods":
      "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers":
      "Content-Type",
  };
}