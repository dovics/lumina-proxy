//! HTTP handlers for proxy endpoints

use super::non_streaming::handle_non_streaming;
use super::responses_convert::{convert_chat_sse_to_responses_sse, fetch_upstream_model_info};
use super::state::ProxyState;
use super::streaming::handle_streaming;
use super::url::{build_backend_url, build_backend_url_for_endpoint};
use crate::config::{Config, ProviderType};
use crate::convert::{convert_chat_to_responses, convert_responses_to_chat};
use crate::stats::StatsWriter;
use crate::token_counter::count_prompt_tokens;
use crate::types::*;
use axum::{
    Json,
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// Main Axum Handler
// =============================================================================

/// Main Axum handler for POST /v1/chat/completions
pub async fn proxy_handler(
    State(state): State<Arc<ProxyState>>,
    bytes: Bytes,
) -> impl IntoResponse {
    if bytes.len() > 100 * 1024 * 1024 {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({ "error": "Request too large, maximum 100MB allowed" })),
        )
            .into_response();
    }

    let req: OpenAIChatRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            let body_snippet = String::from_utf8_lossy(&bytes[..std::cmp::min(500, bytes.len())]);
            tracing::warn!(
                error = %e,
                body_len = bytes.len(),
                body_snippet = %body_snippet,
                "Failed to parse Chat Completions request body"
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("Invalid request body: {}", e) })),
            )
                .into_response();
        }
    };

    tracing::trace!(
        model = %req.model,
        stream = ?req.stream,
        messages_count = %req.messages.len(),
        request_body = %String::from_utf8_lossy(&bytes),
        "Incoming request"
    );

    let config = state.config.load();
    let model = req.model.clone();
    let Some(route) = config.find_backend_for_model(&model) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("No backend route configured for model: {}", model) })),
        )
            .into_response();
    };

    let backend_url = match build_backend_url(route, &model) {
        Ok(url) => url,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };

    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        match handle_streaming(&state, req, route, backend_url).await {
            Ok(response) => response.into_response(),
            Err((status, error)) => (status, error).into_response(),
        }
    } else {
        match handle_non_streaming(&state, req, route, backend_url).await {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err((status, error)) => (status, error).into_response(),
        }
    }
}

// =============================================================================
// Responses API Handler
// =============================================================================

/// Axum handler for POST /v1/responses - OpenAI Responses API endpoint
pub async fn responses_handler(
    State(state): State<Arc<ProxyState>>,
    bytes: Bytes,
) -> impl IntoResponse {
    if bytes.len() > 100 * 1024 * 1024 {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({ "error": "Request too large, maximum 100MB allowed" })),
        )
            .into_response();
    }

    let req: ResponsesRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            let body_snippet = String::from_utf8_lossy(&bytes[..std::cmp::min(500, bytes.len())]);
            tracing::warn!(
                error = %e,
                body_len = bytes.len(),
                body_snippet = %body_snippet,
                "Failed to parse Responses API request body"
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("Invalid request body: {}", e) })),
            )
                .into_response();
        }
    };

    tracing::trace!(
        model = %req.model,
        stream = ?req.stream,
        input_type = ?req.input.as_ref().map(|i| match i {
            ResponseInput::String(_) => "string",
            ResponseInput::Messages(_) => "messages",
            ResponseInput::Raw(_) => "raw",
        }),
        "Incoming Responses API request"
    );

    let config = state.config.load();
    let model = req.model.clone();
    let Some(route) = config.find_backend_for_model(&model) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("No backend route configured for model: {}", model) })),
        )
            .into_response();
    };

    let use_native_responses = matches!(route.provider_type, ProviderType::OpenAi);
    let is_streaming = req.stream.unwrap_or(false);

    if use_native_responses {
        let backend_url = match build_backend_url_for_endpoint(route, &model, true) {
            Ok(url) => url,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e.to_string() })),
                )
                    .into_response();
            }
        };

        let mut request_builder = state.client.post(backend_url);
        if let Some(api_key) = &route.api_key {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", api_key));
        }
        request_builder = request_builder.header("Content-Type", "application/json");

        if is_streaming {
            let response = request_builder.body(bytes.to_vec()).send().await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    let mut headers = HeaderMap::new();
                    headers.insert("Content-Type", "text/event-stream".parse().unwrap());
                    headers.insert("Cache-Control", "no-cache".parse().unwrap());
                    headers.insert("Connection", "keep-alive".parse().unwrap());

                    let body = Body::from_stream(
                        resp.bytes_stream()
                            .map(|result| result.map_err(std::io::Error::other)),
                    );

                    (headers, body).into_response()
                }
                Ok(resp) => {
                    let status = resp.status();
                    let error_text = resp.text().await.unwrap_or_default();
                    (status, Json(json!({ "error": error_text }))).into_response()
                }
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Backend request failed: {}", e) })),
                )
                    .into_response(),
            }
        } else {
            let response = request_builder.body(bytes.to_vec()).send().await;

            match response {
                Ok(resp) if resp.status().is_success() => {
                    match resp.json::<serde_json::Value>().await {
                        Ok(json_body) => (StatusCode::OK, Json(json_body)).into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(json!({ "error": format!("Failed to parse backend response: {}", e) }))
                        ).into_response(),
                    }
                }
                Ok(resp) => {
                    let status = resp.status();
                    let error_text = resp.text().await.unwrap_or_default();
                    (status, Json(json!({ "error": error_text }))).into_response()
                }
                Err(e) => (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Backend request failed: {}", e) })),
                )
                    .into_response(),
            }
        }
    } else {
        let backend_url = match build_backend_url(route, &model) {
            Ok(url) => url,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": e.to_string() })),
                )
                    .into_response();
            }
        };

        let chat_req = convert_responses_to_chat(&req);

        if is_streaming {
            match handle_streaming(&state, chat_req.clone(), route, backend_url).await {
                Ok(response) => {
                    let (parts, body) = response.into_parts();

                    let response_id = format!("resp_{}", std::process::id());
                    let created_at = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    let model = chat_req.model.clone();
                    let prompt_tokens = count_prompt_tokens(&chat_req) as u32;

                    let transformed_stream = convert_chat_sse_to_responses_sse(
                        body,
                        response_id,
                        model,
                        created_at,
                        prompt_tokens,
                    );

                    let mut response = Response::new(Body::from_stream(transformed_stream));
                    *response.status_mut() = parts.status;
                    *response.headers_mut() = parts.headers;
                    response.into_response()
                }
                Err((status, error)) => (status, error).into_response(),
            }
        } else {
            match handle_non_streaming(&state, chat_req, route, backend_url).await {
                Ok(chat_resp) => {
                    let created_at = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    let responses_resp = convert_chat_to_responses(&chat_resp, created_at);
                    (StatusCode::OK, Json(responses_resp)).into_response()
                }
                Err((status, error)) => (status, error).into_response(),
            }
        }
    }
}

// =============================================================================
// Models Handler
// =============================================================================

/// Axum handler for GET /v1/models - returns list of all enabled models
pub async fn models_handler(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let config = state.config.load();
    let mut data = Vec::new();
    for route in config.routes.iter().filter(|route| route.enabled) {
        let model_info = fetch_upstream_model_info(&state.client, route, created).await;
        data.push(model_info);
    }

    let response = OpenAIModelsListResponse {
        object: "list".to_string(),
        data,
    };

    (StatusCode::OK, Json(response)).into_response()
}

// =============================================================================
// Admin API Handlers
// =============================================================================

/// Handler for POST /v1/admin/reload-config
pub async fn reload_config_handler(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let new_config = match Config::load_and_validate(&state.config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::error!("Failed to reload configuration: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "status": "error",
                    "message": format!("Failed to reload configuration: {}", e)
                })),
            )
                .into_response();
        }
    };

    let current_config = state.config.load();

    if new_config.statistics.enabled != current_config.statistics.enabled {
        if new_config.statistics.enabled {
            match StatsWriter::new(&new_config.statistics).await {
                Ok(new_stats_writer) => {
                    state.stats_writer.store(Arc::new(Some(new_stats_writer)));
                    tracing::info!("Statistics enabled during config reload");
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize statistics writer: {}", e);
                }
            }
        } else {
            state.stats_writer.store(Arc::new(None));
            tracing::info!("Statistics disabled during config reload");
        }
    }

    let mut warnings = Vec::new();
    if new_config.server.port != current_config.server.port {
        warnings.push("Server port change requires server restart to take effect".to_string());
    }
    if new_config.server.host != current_config.server.host {
        warnings.push("Server host change requires server restart to take effect".to_string());
    }
    if new_config.logging != current_config.logging {
        warnings.push(
            "Logging configuration changes require server restart to take effect".to_string(),
        );
    }
    if new_config.statistics.stats_file != current_config.statistics.stats_file
        || new_config.statistics.aggregation_interval_secs
            != current_config.statistics.aggregation_interval_secs
    {
        warnings
            .push("Statistics config changes require server restart to take effect.".to_string());
    }
    if new_config.server.proxy != current_config.server.proxy {
        warnings
            .push("Proxy configuration changes require server restart to take effect".to_string());
    }

    state.config.store(Arc::new(new_config));
    tracing::info!("Configuration reloaded successfully");

    (
        StatusCode::OK,
        Json(json!({
            "status": "success",
            "message": "Configuration reloaded successfully",
            "warnings": warnings
        })),
    )
        .into_response()
}

/// Handler for GET /v1/admin/config - returns current configuration (sensitive fields masked)
pub async fn get_config_handler(State(state): State<Arc<ProxyState>>) -> impl IntoResponse {
    let config = state.config.load();

    let routes: Vec<_> = config
        .routes
        .iter()
        .map(|route| {
            json!({
                "model_name": route.model_name,
                "provider_type": format!("{:?}", route.provider_type).to_lowercase(),
                "enabled": route.enabled
            })
        })
        .collect();

    let response = json!({
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "auth_enabled": config.server.auth_token.is_some()
        },
        "logging": {
            "level": config.logging.level,
            "console_enabled": config.logging.console,
            "file_enabled": config.logging.file.as_ref().map(|f| f.enabled).unwrap_or(false)
        },
        "statistics": {
            "enabled": config.statistics.enabled
        },
        "routes": routes,
        "metadata": {
            "version": config.version,
            "loaded_at": config.loaded_at.to_rfc3339()
        }
    });

    (StatusCode::OK, Json(response)).into_response()
}
