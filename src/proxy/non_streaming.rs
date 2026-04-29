//! Non-streaming request handling

use super::state::ProxyState;
use crate::config::{ProviderType, RouteConfig};
use crate::convert::*;
use crate::stats::RequestMetrics;
use crate::token_counter::*;
use crate::types::*;
use axum::{Json, http::StatusCode};
use serde_json::json;

// =============================================================================
// Non-streaming Request Handling
// =============================================================================

/// Handle a non-streaming chat completion request
pub async fn handle_non_streaming(
    state: &ProxyState,
    req: OpenAIChatRequest,
    route: &RouteConfig,
    backend_url: String,
) -> Result<OpenAIChatResponse, (StatusCode, Json<serde_json::Value>)> {
    let model = req.model.clone();
    let start_time = std::time::Instant::now();

    let mut outgoing_req = req;
    let upstream_model = route.upstream_model().to_string();
    outgoing_req.model = upstream_model;

    let prompt_tokens = count_prompt_tokens(&outgoing_req);

    let mut request_builder = state.client.post(backend_url.clone());

    if let Some(api_key) = &route.api_key {
        request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
    }
    request_builder = request_builder.header("Content-Type", "application/json");

    let body = match route.provider_type {
        ProviderType::Ollama => serde_json::to_vec(&convert_openai_to_ollama(&outgoing_req))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to serialize Ollama request: {}", e) })),
                )
            }),

        ProviderType::Anthropic => serde_json::to_vec(&convert_openai_to_anthropic(&outgoing_req))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        json!({ "error": format!("Failed to serialize Anthropic request: {}", e) }),
                    ),
                )
            }),

        ProviderType::Gemini => serde_json::to_vec(&convert_openai_to_gemini(&outgoing_req))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to serialize Gemini request: {}", e) })),
                )
            }),

        ProviderType::OpenAi | ProviderType::OpenAiCompatible => serde_json::to_vec(&outgoing_req)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to serialize OpenAI request: {}", e) })),
                )
            }),

    };

    let body = match body {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    // Debug logging for OpenAI-compatible backends
    if matches!(route.provider_type, ProviderType::OpenAiCompatible)
        && let Ok(body_str) = String::from_utf8(body.clone())
    {
        tracing::debug!(
            "OpenAI-compatible request body: {}",
            if body_str.len() > 2000 {
                // Find a safe char boundary at or before 2000 bytes
                let truncate_at = body_str.char_indices().find(|(i, _)| *i >= 2000).map(|(i, _)| i).unwrap_or(2000);
                format!("{}...", &body_str[..truncate_at])
            } else {
                body_str
            }
        );
    }

    let response = request_builder.body(body).send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Backend request failed: {}", e) })),
        )
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        tracing::error!(
            backend_url = %backend_url,
            status = %status,
            error_response = %error_text,
            "Backend returned error"
        );
        return Err((
            status,
            Json(json!({ "error": format!("Backend returned error: {}", error_text) })),
        ));
    }

    let (openai_resp, completion_tokens) = match route.provider_type {
        ProviderType::Ollama => {
            let ollama_resp: OllamaChatResponse = response.json().await.map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Failed to parse Ollama response: {}", e) })),
                )
            })?;
            let resp = convert_ollama_to_openai(&ollama_resp, &model);
            let tokens = resp.usage.completion_tokens as usize;
            (resp, tokens)
        }

        ProviderType::Anthropic => {
            let anthropic_resp: AnthropicChatResponse = response.json().await.map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Failed to parse Anthropic response: {}", e) })),
                )
            })?;
            let resp = convert_anthropic_to_openai(&anthropic_resp, &model);
            let tokens = resp.usage.completion_tokens as usize;
            (resp, tokens)
        }

        ProviderType::Gemini => {
            let gemini_resp: GeminiChatResponse = response.json().await.map_err(|e| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Failed to parse Gemini response: {}", e) })),
                )
            })?;
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let resp = convert_gemini_to_openai(&gemini_resp, &model, created);
            let tokens = resp.usage.completion_tokens as usize;
            (resp, tokens)
        }

        ProviderType::OpenAi | ProviderType::OpenAiCompatible => {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();

            tracing::debug!(
                status = %status,
                provider = ?route.provider_type,
                body_len = %body_text.len(),
                "Received upstream response body"
            );

            let mut openai_resp: OpenAIChatResponse =
                serde_json::from_str(&body_text).map_err(|e| {
                    tracing::warn!(
                        status = %status,
                        provider = ?route.provider_type,
                        body_preview = %body_text.chars().take(500).collect::<String>(),
                        "Failed to parse OpenAI response: {}", e
                    );
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({ "error": format!("Failed to parse OpenAI response: {}", e) })),
                    )
                })?;
            let tokens = openai_resp.usage.completion_tokens as usize;
            openai_resp.model = model.clone();

            // Normalize content: convert null to empty string for better client compatibility
            // Some providers return content: null when there's no text output (e.g., token limit reached)
            for choice in &mut openai_resp.choices {
                if let Some(ref mut message) = choice.message
                    && message.content.is_none()
                {
                    message.content = Some(MessageContent::String(String::new()));
                }
            }

            (openai_resp, tokens)
        }
    };

    let total_tokens = prompt_tokens + completion_tokens;
    let duration_ms = start_time.elapsed().as_millis() as u64;

    let stats_writer = state.stats_writer.load();
    if let Some(stats_writer) = stats_writer.as_ref() {
        let metric = RequestMetrics {
            model: model.clone(),
            provider: format!("{:?}", route.provider_type).to_lowercase(),
            prompt_tokens: prompt_tokens as u64,
            completion_tokens: completion_tokens as u64,
            duration_ms,
            ttft_ms: None,
            tpot_ms: None,
            status: "success".to_string(),
        };
        if let Err(e) = stats_writer.write_metric(metric).await {
            tracing::warn!("Failed to write statistics: {}", e);
        }
    }

    tracing::info!(
        model = %model,
        provider = %format!("{:?}", route.provider_type),
        prompt_tokens = %prompt_tokens,
        completion_tokens = %completion_tokens,
        total_tokens = %total_tokens,
        duration_ms = %duration_ms,
        "Completed non-streaming request"
    );

    for choice in &openai_resp.choices {
        if let Some(reason) = &choice.finish_reason
            && reason == "length"
        {
            tracing::warn!(
                model = %openai_resp.model,
                finish_reason = %reason,
                "Response was truncated due to token limit (max_tokens or model limit)"
            );
        }
        if let Some(msg) = &choice.message
            && (msg.content.is_none()
                || msg.content.as_ref().map(|c| c.is_empty()).unwrap_or(false))
        {
            tracing::warn!(
                model = %openai_resp.model,
                finish_reason = %choice.finish_reason.clone().unwrap_or_default(),
                "Response has no content"
            );
        }
    }

    tracing::trace!(
        model = %openai_resp.model,
        response_body = %serde_json::to_string(&openai_resp).unwrap_or_default(),
        "Non-streaming response"
    );

    Ok(openai_resp)
}
