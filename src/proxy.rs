//! Core proxy implementation - handles request routing, conversion, and streaming

use crate::config::{Config, ProviderType, RouteConfig};
use crate::convert::*;
use crate::stats::{RequestMetrics, StatsWriter};
use crate::token_counter::*;
use crate::types::*;
use arc_swap::ArcSwap;
use axum::response::Json as AxumJson;
use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::time::Instant;

// =============================================================================
// Streaming Chunk Result - Supports multiple chunks for OpenAI streaming format
// =============================================================================

/// Result type for streaming chunk parsing, supporting multiple chunks for
/// OpenAI-compatible tool call streaming
enum StreamChunkResult {
    /// A single streaming chunk
    Single(OpenAIStreamChunk),
    /// Multiple streaming chunks (used for Moonlight tool calls in OpenAI format)
    Multiple(Vec<OpenAIStreamChunk>),
    /// Parse error
    Error(String),
}

// =============================================================================
// Proxy State - Shared application state
// =============================================================================

/// Shared proxy state that is held by the Axum server
#[derive(Clone)]
pub struct ProxyState {
    /// Application configuration - atomically updatable
    pub config: Arc<ArcSwap<Config>>,
    /// Path to configuration file for reloading
    pub config_path: String,
    /// reqwest HTTP client for backend requests
    pub client: reqwest::Client,
    /// Optional statistics writer for token usage logging - atomically updatable
    pub stats_writer: Arc<ArcSwap<Option<StatsWriter>>>,
}

// =============================================================================
// Backend URL Construction
// =============================================================================

/// Build the appropriate backend URL based on provider type and route config
pub fn build_backend_url(route: &RouteConfig, model: &str) -> Result<String, ProxyError> {
    build_backend_url_for_endpoint(route, model, false)
}

/// Build backend URL with option to use Responses API endpoint
pub fn build_backend_url_for_endpoint(
    route: &RouteConfig,
    model: &str,
    use_responses_api: bool,
) -> Result<String, ProxyError> {
    match route.provider_type {
        ProviderType::Ollama => {
            let base_url = route.base_url.as_ref().ok_or_else(|| {
                ProxyError::ConfigError(format!(
                    "Ollama route for model '{}' missing base_url",
                    model
                ))
            })?;
            Ok(format!("{}/api/chat", base_url.trim_end_matches('/')))
        }

        ProviderType::Anthropic => {
            let base_url = route.base_url.as_ref().ok_or_else(|| {
                ProxyError::ConfigError(format!(
                    "Anthropic route for model '{}' missing base_url",
                    model
                ))
            })?;
            Ok(format!("{}/v1/messages", base_url.trim_end_matches('/')))
        }

        ProviderType::Gemini => {
            let base_url = route.base_url.as_ref().ok_or_else(|| {
                ProxyError::ConfigError(format!(
                    "Gemini route for model '{}' missing base_url",
                    model
                ))
            })?;
            Ok(format!(
                "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
                base_url.trim_end_matches('/'),
                route.upstream_model()
            ))
        }

        ProviderType::OpenAi => {
            if let Some(url) = &route.url {
                // If explicit url is provided, use it directly
                Ok(url.clone())
            } else if let Some(base_url) = &route.base_url {
                let endpoint = if use_responses_api {
                    "/v1/responses"
                } else {
                    "/v1/chat/completions"
                };
                Ok(format!("{}{}", base_url.trim_end_matches('/'), endpoint))
            } else {
                Err(ProxyError::ConfigError(format!(
                    "OpenAI route for model '{}' missing either url or base_url",
                    model
                )))
            }
        }

        ProviderType::OpenAiCompatible => {
            if use_responses_api {
                // For OpenAI-compatible providers, try to use responses endpoint
                // Either append to base_url or use url directly
                if let Some(base_url) = &route.base_url {
                    Ok(format!("{}/v1/responses", base_url.trim_end_matches('/')))
                } else if let Some(url) = &route.url {
                    // Fall back to chat completions if only url is provided
                    Ok(url.clone())
                } else {
                    route.url.clone().ok_or_else(|| {
                        ProxyError::ConfigError(format!(
                            "OpenAI-compatible route for model '{}' missing url",
                            model
                        ))
                    })
                }
            } else {
                route.url.clone().ok_or_else(|| {
                    ProxyError::ConfigError(format!(
                        "OpenAI-compatible route for model '{}' missing url",
                        model
                    ))
                })
            }
        }

        ProviderType::Moonlight => {
            // Moonlight is OpenAI-compatible, use same URL construction
            if let Some(url) = &route.url {
                Ok(url.clone())
            } else if let Some(base_url) = &route.base_url {
                let endpoint = if use_responses_api {
                    "/v1/responses"
                } else {
                    "/v1/chat/completions"
                };
                Ok(format!("{}{}", base_url.trim_end_matches('/'), endpoint))
            } else {
                Err(ProxyError::ConfigError(format!(
                    "Moonlight route for model '{}' missing either url or base_url",
                    model
                )))
            }
        }
    }
}

// =============================================================================
// Tool Call Aggregation Helpers
// =============================================================================

/// Aggregates streaming tool_call chunks into a complete tool_call
/// Tool calls come in pieces: id, function.name, function.arguments across multiple chunks
fn aggregate_tool_calls(tool_calls: &[OpenAIToolCall]) -> Vec<OpenAIToolCall> {
    use std::collections::HashMap;

    let mut aggregated: HashMap<u32, OpenAIToolCall> = HashMap::new();

    for tc in tool_calls {
        let index = tc.index.unwrap_or(0);
        let entry = aggregated.entry(index).or_insert_with(|| OpenAIToolCall {
            id: None,
            index: Some(index),
            r#type: tc.r#type.clone(),
            function: None,
        });

        if let Some(ref func) = tc.function {
            let func_entry = entry.function.get_or_insert(OpenAIToolCallFunction {
                name: None,
                arguments: None,
            });
            if func.name.is_some() {
                func_entry.name = func.name.clone();
            }
            if func.arguments.is_some() {
                // Concatenate arguments (they come in pieces)
                let new_arg = func.arguments.clone().unwrap_or_default();
                func_entry.arguments =
                    Some(func_entry.arguments.clone().unwrap_or_default() + &new_arg);
            }
        }

        if tc.id.is_some() {
            entry.id = tc.id.clone();
        }
    }

    aggregated.into_values().collect()
}

/// Parse tool calls from Moonlight's special marker format embedded in content field.
///
/// Marker format:
/// - `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>` - wraps all tool calls
/// - `<|tool_call_begin|>` / `<|tool_call_end|>` - wraps each tool call
/// - `<|tool_call_argument_begin|>` - separates tool ID from arguments
///
/// Tool ID format: `functions.{func_name}:{idx}`
pub fn parse_moonlight_tool_calls(content: &str) -> Vec<OpenAIToolCall> {
    let mut tool_calls = Vec::new();

    // Find the tool calls section
    let section_start = match content.find("<|tool_calls_section_begin|>") {
        Some(pos) => pos + "<|tool_calls_section_begin|>".len(),
        None => return tool_calls, // No tool calls section
    };

    let section_end = match content.find("<|tool_calls_section_end|>") {
        Some(pos) => pos,
        None => return tool_calls, // Malformed, no end marker
    };

    let section_content = &content[section_start..section_end];
    let mut index: u32 = 0;

    // Parse each tool call
    let mut remaining = section_content;
    while let Some(call_start) = remaining.find("<|tool_call_begin|>") {
        let after_call_start = &remaining[call_start + "<|tool_call_begin|>".len()..];
        if let Some(call_end) = after_call_start.find("<|tool_call_end|>") {
            let call_content = &after_call_start[..call_end];
            if let Some(tool_call) = extract_moonlight_tool_call(call_content, index) {
                tool_calls.push(tool_call);
                index += 1;
            }
            remaining = &after_call_start[call_end + "<|tool_call_end|>".len()..];
        } else {
            break;
        }
    }

    tool_calls
}

/// Extract a single tool call from a Moonlight tool call segment
fn extract_moonlight_tool_call(segment: &str, index: u32) -> Option<OpenAIToolCall> {
    let parts: Vec<&str> = segment.split("<|tool_call_argument_begin|>").collect();
    if parts.len() != 2 {
        tracing::warn!(
            "Malformed tool call segment: expected 2 parts separated by <|tool_call_argument_begin|>"
        );
        return None;
    }

    let tool_id = parts[0].trim();
    let arguments = parts[1].trim();

    // Parse function name from tool ID (format: functions.{func_name}:{idx})
    let func_name = if let Some(stripped) = tool_id.strip_prefix("functions.") {
        match stripped.rfind(':') {
            Some(pos) => &stripped[..pos],
            None => {
                tracing::warn!("Malformed tool ID '{}': missing colon separator", tool_id);
                return None;
            }
        }
    } else {
        tracing::warn!(
            "Malformed tool ID '{}': missing 'functions.' prefix",
            tool_id
        );
        return None;
    };

    // Generate a random ID
    let id = format!("call_{}", generate_random_id());

    Some(OpenAIToolCall {
        index: Some(index),
        id: Some(id),
        r#type: Some("function".to_string()), // type is reserved keyword in Rust
        function: Some(OpenAIToolCallFunction {
            name: Some(func_name.to_string()),
            arguments: Some(arguments.to_string()),
        }),
    })
}

/// Generate a random 12-character alphanumeric ID
fn generate_random_id() -> String {
    use rand::Rng;
    use std::iter;
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::thread_rng();
    iter::repeat_with(|| {
        let idx = rng.gen_range(0..CHARSET.len());
        CHARSET[idx] as char
    })
    .take(12)
    .collect()
}

/// Strip Moonlight tool call markers from content, preserving surrounding text
fn strip_moonlight_tool_markers(content: &str) -> String {
    let markers = [
        "<|tool_calls_section_begin|>",
        "<|tool_calls_section_end|>",
        "<|tool_call_begin|>",
        "<|tool_call_end|>",
        "<|tool_call_argument_begin|>",
    ];
    let mut result = content.to_string();
    for marker in &markers {
        result = result.replace(marker, "");
    }
    result
}

// =============================================================================
// Non-streaming Request Handling
// =============================================================================

/// Handle a non-streaming chat completion request
async fn handle_non_streaming(
    state: &ProxyState,
    req: OpenAIChatRequest,
    route: &RouteConfig,
    backend_url: String,
) -> Result<OpenAIChatResponse, (StatusCode, Json<serde_json::Value>)> {
    let model = req.model.clone();
    let start_time = std::time::Instant::now();

    // Use upstream_model if configured for outgoing request
    let mut outgoing_req = req;
    let upstream_model = route.upstream_model().to_string();
    outgoing_req.model = upstream_model;

    // Spawn token counting in a separate task
    let prompt_tokens = count_prompt_tokens(&outgoing_req);

    // Build the request based on provider type
    let mut request_builder = state.client.post(backend_url);

    // Add authorization header if api_key is provided
    if let Some(api_key) = &route.api_key {
        request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
    }
    // Set content type to JSON since we're sending JSON body
    request_builder = request_builder.header("Content-Type", "application/json");

    // Convert request body based on provider
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

        ProviderType::Moonlight => serde_json::to_vec(&convert_openai_to_moonlight(&outgoing_req))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        json!({ "error": format!("Failed to serialize Moonlight request: {}", e) }),
                    ),
                )
            }),
    };

    let body = match body {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    // Send the request
    let response = request_builder.body(body).send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Backend request failed: {}", e) })),
        )
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err((
            status,
            Json(json!({ "error": format!("Backend returned error: {}", error_text) })),
        ));
    }

    // Get completion tokens based on provider
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
            // Override model name to client-requested name for response
            openai_resp.model = model.clone();
            (openai_resp, tokens)
        }

        ProviderType::Moonlight => {
            // Moonlight is OpenAI-compatible, handle like OpenAI
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
                        "Failed to parse Moonlight response: {}", e
                    );
                    (
                        StatusCode::BAD_GATEWAY,
                        Json(json!({ "error": format!("Failed to parse Moonlight response: {}", e) })),
                    )
                })?;
            let tokens = openai_resp.usage.completion_tokens as usize;
            // Override model name to client-requested name for response
            openai_resp.model = model.clone();
            (openai_resp, tokens)
        }
    };

    let total_tokens = prompt_tokens + completion_tokens;
    let duration_ms = start_time.elapsed().as_millis() as u64;

    // Write statistics if enabled
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

    // Check for abnormal responses and warn
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

    // Trace log the response
    tracing::trace!(
        model = %openai_resp.model,
        response_body = %serde_json::to_string(&openai_resp).unwrap_or_default(),
        "Non-streaming response"
    );

    Ok(openai_resp)
}

// =============================================================================
// Streaming Request Handling
// =============================================================================

/// Handle a streaming chat completion request
async fn handle_streaming(
    state: &ProxyState,
    req: OpenAIChatRequest,
    route: &RouteConfig,
    backend_url: String,
) -> Result<Response<Body>, (StatusCode, Json<serde_json::Value>)> {
    let model = req.model.clone();
    let start_time = std::time::Instant::now();
    let first_bytes_time = Arc::new(Mutex::new(None::<Instant>));

    // Convert request body based on provider - use upstream_model for outgoing request
    let mut streaming_req = req.clone();
    streaming_req.stream = Some(true);
    streaming_req.model = route.upstream_model().to_string();

    // Count prompt tokens synchronously (fast enough) - use upstream model for correct tokenizer selection
    let prompt_tokens = count_prompt_tokens(&streaming_req);

    let body = match route.provider_type {
        ProviderType::Ollama => serde_json::to_vec(&convert_openai_to_ollama(&streaming_req))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to serialize Ollama request: {}", e) })),
                )
            }),

        ProviderType::Anthropic => serde_json::to_vec(&convert_openai_to_anthropic(&streaming_req))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        json!({ "error": format!("Failed to serialize Anthropic request: {}", e) }),
                    ),
                )
            }),

        ProviderType::Gemini => serde_json::to_vec(&convert_openai_to_gemini(&streaming_req))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to serialize Gemini request: {}", e) })),
                )
            }),

        ProviderType::OpenAi | ProviderType::OpenAiCompatible => serde_json::to_vec(&streaming_req)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({ "error": format!("Failed to serialize OpenAI request: {}", e) })),
                )
            }),

        ProviderType::Moonlight => serde_json::to_vec(&convert_openai_to_moonlight(&streaming_req))
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(
                        json!({ "error": format!("Failed to serialize Moonlight request: {}", e) }),
                    ),
                )
            }),
    };

    let body = match body {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    // Build and send the request
    let mut request_builder = state.client.post(backend_url.clone());
    if let Some(api_key) = &route.api_key {
        request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
    }
    // Set content type to JSON since we're sending JSON body
    request_builder = request_builder.header("Content-Type", "application/json");

    // Some providers require specific headers for SSE
    request_builder = request_builder.header("Accept", "text/event-stream");

    let response = request_builder.body(body).send().await.map_err(|e| {
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Backend request failed: {}", e) })),
        )
    })?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err((
            status,
            Json(json!({ "error": format!("Backend returned error: {}", error_text) })),
        ));
    }

    // Get the response stream
    let bytes_stream = response.bytes_stream();

    // Create shared token counter for incremental counting
    let token_counter = Arc::new(IncrementalTokenCounter::new());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let response_id = format!(
        "{}-{}",
        match route.provider_type {
            ProviderType::OpenAi => "chatcmpl",
            ProviderType::Ollama => "ollama",
            ProviderType::Anthropic => "anthropic",
            ProviderType::Gemini => "gemini",
            ProviderType::Moonlight => "moonlight",
            ProviderType::OpenAiCompatible => "chatcmpl",
        },
        created
    );

    let provider_type = route.provider_type;
    let model_clone = model.clone();
    let stats_writer_snapshot = state.stats_writer.load();
    let token_counter_clone = token_counter.clone();

    // Transform the stream: parse each chunk, convert to OpenAI format, count tokens
    // Use unfold to maintain buffer state between chunks
    // Pin the incoming stream because we need it across multiple async calls
    let pinned_bytes_stream = Box::pin(bytes_stream);
    let initial_state = (
        pinned_bytes_stream,
        token_counter_clone,
        response_id.clone(),
        created,
        model_clone,
        String::new(),
        first_bytes_time.clone(),
        true, // is_first_chunk - we'll add usage to the first successful chunk
    );

    // Clone prompt_tokens for the transformed stream (moved into closure)
    let prompt_tokens_stream = prompt_tokens;
    let transformed_stream = futures_util::stream::unfold(
        initial_state,
        move |(
            mut bytes_stream,
            counter,
            id,
            created,
            model,
            mut buffer,
            first_bytes_time,
            is_first_chunk,
        )| async move {
            let provider_type = provider_type;

            // Continuously read and process until we have at least one chunk to yield,
            // or the backend stream ends
            loop {
                // Process complete lines from buffer first (we might have leftover from last iteration)
                let mut yielded_chunks = Vec::new();
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[0..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() || !line.starts_with("data: ") {
                        continue;
                    }

                    let data = &line["data: ".len()..];

                    // Log every non-DONE chunk for debugging
                    if data.trim() != "[DONE]" {
                        tracing::trace!(
                            "Received SSE chunk: {}",
                            if data.len() < 500 {
                                data
                            } else {
                                "<chunk too long>"
                            }
                        );
                    }

                    // Skip [DONE] message for now, we'll add it at the end
                    if data.trim() == "[DONE]" {
                        continue;
                    }

                    // Parse and convert chunk based on provider
                    let openai_chunk_result: StreamChunkResult = match provider_type {
                        ProviderType::Ollama => {
                            match serde_json::from_str::<OllamaStreamChunk>(data) {
                                Ok(ollama_chunk) => {
                                    // Count tokens incrementally
                                    if let Some(delta) = &ollama_chunk.delta {
                                        counter.add_delta(&delta.content);
                                    }
                                    StreamChunkResult::Single(
                                        convert_ollama_stream_chunk_to_openai(
                                            &ollama_chunk,
                                            &id,
                                            created,
                                            &model,
                                        ),
                                    )
                                }
                                Err(e) => StreamChunkResult::Error(format!(
                                    "Failed to parse Ollama stream chunk: {}",
                                    e
                                )),
                            }
                        }

                        ProviderType::Anthropic => {
                            match serde_json::from_str::<AnthropicStreamChunk>(data) {
                                Ok(anthropic_chunk) => {
                                    if let Some(delta) = &anthropic_chunk.delta
                                        && let Some(text) = &delta.text
                                    {
                                        counter.add_delta(text);
                                    }
                                    StreamChunkResult::Single(
                                        convert_anthropic_stream_chunk_to_openai(
                                            &anthropic_chunk,
                                            created,
                                            &model,
                                        ),
                                    )
                                }
                                Err(e) => StreamChunkResult::Error(format!(
                                    "Failed to parse Anthropic stream chunk: {}",
                                    e
                                )),
                            }
                        }

                        ProviderType::Gemini => {
                            // Gemini SSE doesn't always use "data: " prefix, but we already handled that
                            match serde_json::from_str::<GeminiStreamChunk>(data) {
                                Ok(gemini_chunk) => {
                                    // Count tokens from all candidates
                                    for candidate in &gemini_chunk.candidates {
                                        for part in &candidate.content.parts {
                                            if let Some(text) = &part.text {
                                                counter.add_delta(text);
                                            }
                                        }
                                    }
                                    StreamChunkResult::Single(
                                        convert_gemini_stream_chunk_to_openai(
                                            &gemini_chunk,
                                            &id,
                                            created,
                                            &model,
                                        ),
                                    )
                                }
                                Err(e) => StreamChunkResult::Error(format!(
                                    "Failed to parse Gemini stream chunk: {}",
                                    e
                                )),
                            }
                        }

                        ProviderType::OpenAi | ProviderType::OpenAiCompatible => {
                            // First, try to parse as raw JSON to see the actual structure
                            let raw_value: serde_json::Value = match serde_json::from_str(data) {
                                Ok(v) => v,
                                Err(_) => {
                                    // If not valid JSON, just try the original parse
                                    serde_json::Value::Null
                                }
                            };

                            match serde_json::from_str::<OpenAIStreamChunk>(data) {
                                Ok(mut openai_chunk) => {
                                    let mut has_content = false;
                                    // Count tokens from all choices
                                    for choice in &openai_chunk.choices {
                                        // Count content tokens
                                        if let Some(content) = &choice.delta.content
                                            && !content.is_empty()
                                        {
                                            counter.add_delta(content);
                                            has_content = true;
                                        }
                                        // Also count reasoning tokens (for Kimi/OpenRouter deepseek reasoning)
                                        if let Some(reasoning) = &choice.delta.reasoning
                                            && !reasoning.is_empty()
                                        {
                                            counter.add_delta(reasoning);
                                            has_content = true;
                                        }
                                        // Also count tool call tokens (function name and arguments)
                                        if let Some(tool_calls) = &choice.delta.tool_calls {
                                            for tool_call in tool_calls {
                                                if let Some(function) = &tool_call.function {
                                                    if let Some(name) = &function.name
                                                        && !name.is_empty()
                                                    {
                                                        counter.add_delta(name);
                                                        has_content = true;
                                                    }
                                                    if let Some(arguments) = &function.arguments
                                                        && !arguments.is_empty()
                                                    {
                                                        counter.add_delta(arguments);
                                                        has_content = true;
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    // Aggregate and forward tool_calls delta to client
                                    for choice in &mut openai_chunk.choices {
                                        if let Some(ref tool_calls) = choice.delta.tool_calls {
                                            let aggregated = aggregate_tool_calls(tool_calls);
                                            if !aggregated.is_empty() {
                                                choice.delta.tool_calls = Some(aggregated);
                                            }
                                        }
                                    }

                                    // Debug: if chunk parsed but no countable content, log at trace level
                                    if !has_content {
                                        tracing::trace!(
                                            "Chunk parsed but no countable content: {}",
                                            serde_json::to_string(&raw_value)
                                                .unwrap_or_else(|_| data.to_string())
                                        );
                                    }

                                    // Ensure the response ID is consistent
                                    openai_chunk.id = id.clone();
                                    // Override model name to client-requested name
                                    openai_chunk.model = model.clone();
                                    StreamChunkResult::Single(openai_chunk)
                                }
                                Err(e) => {
                                    // Log raw data when parse fails for debugging
                                    tracing::debug!(
                                        "Failed to parse OpenAI stream chunk: {}, raw data: {}",
                                        e,
                                        if data.len() < 1000 {
                                            data
                                        } else {
                                            "data too long, see trace"
                                        }
                                    );
                                    StreamChunkResult::Error(format!(
                                        "Failed to parse OpenAI stream chunk: {}",
                                        e
                                    ))
                                }
                            }
                        }

                        ProviderType::Moonlight => {
                            // Moonlight: parse tool calls from content markers
                            // Output as OpenAI streaming SSE format
                            match serde_json::from_str::<OpenAIStreamChunk>(data) {
                                Ok(mut openai_chunk) => {
                                    let mut has_content = false;
                                    let mut tool_call_chunks: Vec<OpenAIStreamChunk> = Vec::new();

                                    for choice in &mut openai_chunk.choices {
                                        if let Some(content) = &choice.delta.content {
                                            let content_str = content.as_str();

                                            // Check for tool call section markers
                                            if content_str.contains("<|tool_calls_section_begin|>")
                                            {
                                                let parsed =
                                                    parse_moonlight_tool_calls(content_str);
                                                let text_only =
                                                    strip_moonlight_tool_markers(content_str);

                                                if !parsed.is_empty() {
                                                    // Count tokens
                                                    for tc in &parsed {
                                                        if let Some(ref f) = tc.function {
                                                            if let Some(ref n) = f.name {
                                                                counter.add_delta(n);
                                                            }
                                                            if let Some(ref a) = f.arguments {
                                                                counter.add_delta(a);
                                                            }
                                                        }
                                                    }

                                                    // Create OpenAI streaming chunks for each tool call
                                                    // First chunk: role + tool_calls with first tool call
                                                    let first_tc = &parsed[0];
                                                    let first_chunk = OpenAIStreamChunk {
                                                        id: id.clone(),
                                                        object: "chat.completion.chunk".to_string(),
                                                        created,
                                                        model: model.clone(),
                                                        choices: vec![OpenAIStreamChoice {
                                                            index: 0,
                                                            delta: OpenAIDelta {
                                                                role: Some("assistant".to_string()),
                                                                content: if text_only
                                                                    .trim()
                                                                    .is_empty()
                                                                {
                                                                    None
                                                                } else {
                                                                    Some(text_only.clone())
                                                                },
                                                                reasoning: None,
                                                                tool_calls: Some(vec![
                                                                    OpenAIToolCall {
                                                                        index: first_tc.index,
                                                                        id: first_tc.id.clone(),
                                                                        r#type: first_tc
                                                                            .r#type
                                                                            .clone(),
                                                                        function: first_tc
                                                                            .function
                                                                            .clone(),
                                                                    },
                                                                ]),
                                                            },
                                                            finish_reason: None,
                                                        }],
                                                        usage: None,
                                                    };
                                                    tool_call_chunks.push(first_chunk);

                                                    // Subsequent tool calls (index > 0) as separate chunks
                                                    for tc in parsed.iter().skip(1) {
                                                        let chunk = OpenAIStreamChunk {
                                                            id: id.clone(),
                                                            object: "chat.completion.chunk"
                                                                .to_string(),
                                                            created,
                                                            model: model.clone(),
                                                            choices: vec![OpenAIStreamChoice {
                                                                index: 0,
                                                                delta: OpenAIDelta {
                                                                    role: None,
                                                                    content: None,
                                                                    reasoning: None,
                                                                    tool_calls: Some(vec![
                                                                        OpenAIToolCall {
                                                                            index: tc.index,
                                                                            id: tc.id.clone(),
                                                                            r#type: tc
                                                                                .r#type
                                                                                .clone(),
                                                                            function: tc
                                                                                .function
                                                                                .clone(),
                                                                        },
                                                                    ]),
                                                                },
                                                                finish_reason: None,
                                                            }],
                                                            usage: None,
                                                        };
                                                        tool_call_chunks.push(chunk);
                                                    }

                                                    // Final chunk: finish_reason = "tool_calls"
                                                    let final_chunk = OpenAIStreamChunk {
                                                        id: id.clone(),
                                                        object: "chat.completion.chunk".to_string(),
                                                        created,
                                                        model: model.clone(),
                                                        choices: vec![OpenAIStreamChoice {
                                                            index: 0,
                                                            delta: OpenAIDelta {
                                                                role: None,
                                                                content: None,
                                                                reasoning: None,
                                                                tool_calls: None,
                                                            },
                                                            finish_reason: Some(
                                                                "tool_calls".to_string(),
                                                            ),
                                                        }],
                                                        usage: None,
                                                    };
                                                    tool_call_chunks.push(final_chunk);

                                                    has_content = true;
                                                } else if !text_only.is_empty() {
                                                    counter.add_delta(&text_only);
                                                    choice.delta.content = Some(text_only);
                                                    has_content = true;
                                                }
                                            } else {
                                                counter.add_delta(content_str);
                                                has_content = true;
                                            }
                                        }
                                    }

                                    if !has_content {
                                        tracing::trace!("Moonlight chunk: {}", data.len().min(200));
                                    }

                                    if !tool_call_chunks.is_empty() {
                                        // Return multiple chunks in OpenAI streaming format
                                        StreamChunkResult::Multiple(tool_call_chunks)
                                    } else {
                                        openai_chunk.id = id.clone();
                                        openai_chunk.model = model.clone();
                                        StreamChunkResult::Single(openai_chunk)
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to parse Moonlight stream chunk: {}", e);
                                    StreamChunkResult::Error(format!(
                                        "Failed to parse Moonlight stream chunk: {}",
                                        e
                                    ))
                                }
                            }
                        }
                    };

                    match openai_chunk_result {
                        StreamChunkResult::Single(mut chunk) => {
                            // Add usage to first chunk for clients that expect prompt_tokens (e.g., litellm)
                            if is_first_chunk {
                                chunk.usage = Some(OpenAIUsage {
                                    prompt_tokens: prompt_tokens_stream as u32,
                                    completion_tokens: 0,
                                    total_tokens: prompt_tokens_stream as u32,
                                });
                            }

                            // Trace log each streaming chunk
                            let first_delta = chunk
                                .choices
                                .first()
                                .and_then(|c| c.delta.content.clone())
                                .unwrap_or_default();
                            tracing::trace!(
                                chunk_id = %chunk.id,
                                chunk_model = %chunk.model,
                                first_delta = %first_delta,
                                "Streaming chunk"
                            );

                            // Format as SSE
                            let sse_line =
                                format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                            yielded_chunks.push(Ok(Bytes::from(sse_line)));
                        }
                        StreamChunkResult::Multiple(mut chunks) => {
                            // Add usage to first chunk for clients that expect prompt_tokens
                            if is_first_chunk && let Some(first) = chunks.first_mut() {
                                first.usage = Some(OpenAIUsage {
                                    prompt_tokens: prompt_tokens_stream as u32,
                                    completion_tokens: 0,
                                    total_tokens: prompt_tokens_stream as u32,
                                });
                            }

                            // Push all chunks as SSE
                            for chunk in chunks {
                                tracing::trace!(
                                    chunk_id = %chunk.id,
                                    chunk_model = %chunk.model,
                                    "Streaming chunk (tool_call)"
                                );
                                let sse_line =
                                    format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                                yielded_chunks.push(Ok(Bytes::from(sse_line)));
                            }
                        }
                        StreamChunkResult::Error(e) => {
                            tracing::warn!("{}", e);
                            // Send parse error to client as a warning chunk
                            let error_chunk = serde_json::json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": null
                                }],
                                "warning": e
                            });
                            let sse_line = format!(
                                "data: {}\n\n",
                                serde_json::to_string(&error_chunk).unwrap()
                            );
                            yielded_chunks.push(Ok(Bytes::from(sse_line)));
                        }
                    }
                }

                // If we have chunks to yield, return the first one now
                if !yielded_chunks.is_empty() {
                    let next_chunk = yielded_chunks.remove(0);
                    // Any extra chunks get added back to the buffer
                    for bytes in yielded_chunks.into_iter().flatten() {
                        let s = String::from_utf8_lossy(bytes.as_ref());
                        buffer = s.to_string() + &buffer;
                    }
                    return Some((
                        next_chunk,
                        (
                            bytes_stream,
                            counter,
                            id,
                            created,
                            model,
                            buffer,
                            first_bytes_time,
                            false,
                        ),
                    ));
                }

                // No chunks ready yet - need to read more data from backend
                match bytes_stream.next().await {
                    Some(bytes_result) => {
                        let bytes = match bytes_result {
                            Ok(b) => b,
                            Err(e) => {
                                tracing::error!("Stream error: {}", e);
                                let error_chunk: Result<Bytes, reqwest::Error> =
                                    Ok(Bytes::from(format!("data: {{\"error\": \"{}\"}}\n\n", e)));
                                return Some((
                                    error_chunk,
                                    (
                                        bytes_stream,
                                        counter,
                                        id,
                                        created,
                                        model,
                                        buffer,
                                        first_bytes_time,
                                        false,
                                    ),
                                ));
                            }
                        };

                        // Record TTFT clock: first bytes received from upstream
                        {
                            let mut first_time = first_bytes_time.lock().await;
                            if first_time.is_none() {
                                *first_time = Some(Instant::now());
                            }
                        }

                        // Convert bytes to string and append to buffer
                        match String::from_utf8(bytes.to_vec()) {
                            Ok(s) => buffer.push_str(&s),
                            Err(_) => {
                                tracing::warn!("Received invalid UTF-8 in stream");
                            }
                        }
                        // Continue the loop to process the new buffer content
                    }
                    None => {
                        // Backend stream truly ended
                        if !buffer.is_empty() {
                            tracing::debug!(
                                "Stream ended with unprocessed buffer content: {} bytes",
                                buffer.len()
                            );
                            if buffer.len() < 2000 {
                                tracing::debug!("Unprocessed buffer: {}", buffer);
                            }
                        }
                        return None;
                    }
                }
            }
        },
    );

    // The stream is already okay because Box::pin gives Unpin

    // After stream completes, write statistics
    // Pin the transformed stream so we can safely poll it
    let pinned_transformed = Box::pin(transformed_stream);
    let model_clone = model.clone();
    let provider_type_clone = provider_type;
    let token_counter_clone = token_counter.clone();

    let final_stream = futures_util::stream::unfold(
        (
            pinned_transformed,
            stats_writer_snapshot,
            model_clone,
            provider_type_clone,
            prompt_tokens,
            start_time,
            token_counter_clone,
            first_bytes_time,
            false,
            0, // chunk counter - detect if we got any valid chunks
        ),
        |(
            mut stream,
            stats_writer,
            model,
            provider_type,
            prompt_tokens,
            start_time,
            token_counter_clone,
            first_bytes_time,
            done,
            chunk_count,
        )| async move {
            if done {
                return None;
            }

            // Use poll_next with pinned projection to avoid Unpin issue
            use futures_util::StreamExt;
            match stream.next().await {
                Some(item) => {
                    // Got a valid chunk, increment counter
                    Some((
                        item,
                        (
                            stream,
                            stats_writer,
                            model,
                            provider_type,
                            prompt_tokens,
                            start_time,
                            token_counter_clone,
                            first_bytes_time,
                            false,
                            chunk_count + 1,
                        ),
                    ))
                }
                None => {
                    // Stream is complete, check if we got any valid chunks
                    let completion_tokens = token_counter_clone.total();
                    let total_tokens = prompt_tokens + completion_tokens;
                    let duration_ms = start_time.elapsed().as_millis() as u64;

                    // If no completion tokens were counted, log detailed debugging info
                    if completion_tokens == 0 {
                        tracing::warn!(
                            model = %model,
                            provider = %format!("{:?}", provider_type),
                            duration_ms = %duration_ms,
                            chunks_received = %chunk_count,
                            prompt_tokens = %prompt_tokens,
                            "Zero completion tokens detected - possible tool calls or empty response"
                        );
                    }

                    // If no chunks were received and no completion tokens were counted,
                    // the model returned an empty response - send an error first
                    if chunk_count == 0 {
                        tracing::error!(
                            model = %model,
                            provider = %format!("{:?}", provider_type),
                            duration_ms = %duration_ms,
                            prompt_tokens = %prompt_tokens,
                            "CRITICAL: Model returned no response chunks at all - upstream may have failed"
                        );

                        let error_chunk = serde_json::json!({
                            "id": format!("chatcmpl-{}", std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0)),
                            "object": "chat.completion.chunk",
                            "created": std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_secs())
                                .unwrap_or(0),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": "The model returned no response. This can happen if the upstream service had an error or returned an empty response."
                                },
                                "finish_reason": "stop"
                            }]
                        });

                        let error_sse =
                            format!("data: {}\n\n", serde_json::to_string(&error_chunk).unwrap());

                        // Write stats with error status
                        if let Some(stats_writer) = stats_writer.as_ref() {
                            let metric = RequestMetrics {
                                model: model.clone(),
                                provider: format!("{:?}", provider_type).to_lowercase(),
                                prompt_tokens: prompt_tokens as u64,
                                completion_tokens: completion_tokens as u64,
                                duration_ms,
                                ttft_ms: None,
                                tpot_ms: None,
                                status: "error".to_string(),
                            };
                            if let Err(e) = stats_writer.write_metric(metric).await {
                                tracing::warn!("Failed to write statistics: {}", e);
                            }
                        }

                        // Send error chunk first, then [DONE]
                        return Some((
                            Ok(Bytes::from(error_sse)),
                            (
                                stream,
                                stats_writer,
                                model,
                                provider_type,
                                prompt_tokens,
                                start_time,
                                token_counter_clone,
                                first_bytes_time,
                                true,
                                chunk_count,
                            ),
                        ));
                    }

                    tracing::info!(
                        model = %model,
                        provider = %format!("{:?}", provider_type),
                        prompt_tokens = %prompt_tokens,
                        completion_tokens = %completion_tokens,
                        total_tokens = %total_tokens,
                        duration_ms = %duration_ms,
                        chunks = %chunk_count,
                        "Completed streaming request"
                    );

                    // Calculate TTFT and TPOT
                    let first_bytes_guard = first_bytes_time.lock().await;
                    let ttft_ms = first_bytes_guard.map(|t| t.elapsed().as_millis() as u64);
                    let tpot_ms = if completion_tokens > 0 {
                        Some(duration_ms as f64 / completion_tokens as f64)
                    } else {
                        None
                    };
                    drop(first_bytes_guard);

                    if let Some(stats_writer) = stats_writer.as_ref() {
                        let metric = RequestMetrics {
                            model: model.clone(),
                            provider: format!("{:?}", provider_type).to_lowercase(),
                            prompt_tokens: prompt_tokens as u64,
                            completion_tokens: completion_tokens as u64,
                            duration_ms,
                            ttft_ms,
                            tpot_ms,
                            status: "success".to_string(),
                        };
                        if let Err(e) = stats_writer.write_metric(metric).await {
                            tracing::warn!("Failed to write statistics: {}", e);
                        }
                    }

                    // Send final [DONE]
                    Some((
                        Ok(Bytes::from("data: [DONE]\n\n")),
                        (
                            stream,
                            stats_writer,
                            model,
                            provider_type,
                            prompt_tokens,
                            start_time,
                            token_counter_clone,
                            first_bytes_time,
                            true,
                            chunk_count,
                        ),
                    ))
                }
            }
        },
    );

    let body = Body::from_stream(final_stream);

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "text/event-stream".parse().unwrap());
    headers.insert("Cache-Control", "no-cache".parse().unwrap());
    headers.insert("Connection", "keep-alive".parse().unwrap());

    Ok((headers, body).into_response())
}

// =============================================================================
// Main Axum Handler
// =============================================================================

/// Main Axum handler for POST /v1/chat/completions
pub async fn proxy_handler(
    State(state): State<Arc<ProxyState>>,
    bytes: Bytes,
) -> impl IntoResponse {
    // Check content length limit (100MB)
    if bytes.len() > 100 * 1024 * 1024 {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({ "error": "Request too large, maximum 100MB allowed" })),
        )
            .into_response();
    }
    // Parse OpenAI request from body
    let req: OpenAIChatRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            // Log the error with a snippet of the request body for debugging
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

    // Trace log the incoming request
    tracing::trace!(
        model = %req.model,
        stream = ?req.stream,
        messages_count = %req.messages.len(),
        request_body = %String::from_utf8_lossy(&bytes),
        "Incoming request"
    );

    // Find backend route for requested model
    let config = state.config.load();
    let model = req.model.clone();
    let Some(route) = config.find_backend_for_model(&model) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("No backend route configured for model: {}", model) })),
        )
            .into_response();
    };

    // Build backend URL
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

    // Dispatch based on streaming flag
    let is_streaming = req.stream.unwrap_or(false);

    if is_streaming {
        match handle_streaming(&state, req, route, backend_url).await {
            Ok(response) => response.into_response(),
            Err((status, error)) => (status, error).into_response(),
        }
    } else {
        match handle_non_streaming(&state, req, route, backend_url).await {
            Ok(response) => (StatusCode::OK, AxumJson(response)).into_response(),
            Err((status, error)) => (status, error).into_response(),
        }
    }
}

/// Axum handler for POST /v1/responses - OpenAI Responses API endpoint
pub async fn responses_handler(
    State(state): State<Arc<ProxyState>>,
    bytes: Bytes,
) -> impl IntoResponse {
    // Check content length limit (100MB)
    if bytes.len() > 100 * 1024 * 1024 {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(json!({ "error": "Request too large, maximum 100MB allowed" })),
        )
            .into_response();
    }

    // Parse Responses API request from body
    let req: ResponsesRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            // Log the error with a snippet of the request body for debugging
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

    // Trace log the incoming request
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

    // Find backend route for requested model
    let config = state.config.load();
    let model = req.model.clone();
    let Some(route) = config.find_backend_for_model(&model) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("No backend route configured for model: {}", model) })),
        )
            .into_response();
    };

    // Determine if we should use native Responses API or convert to Chat Completions
    // Only OpenAI gets native Responses API, all others get conversion to Chat Completions
    // OpenAiCompatible backends may not support the Responses API format, so convert
    let use_native_responses = matches!(route.provider_type, ProviderType::OpenAi);

    let is_streaming = req.stream.unwrap_or(false);

    if use_native_responses {
        // Passthrough mode: send directly to /v1/responses endpoint
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

        // Use the raw request body for passthrough
        let mut request_builder = state.client.post(backend_url);
        if let Some(api_key) = &route.api_key {
            request_builder =
                request_builder.header("Authorization", format!("Bearer {}", api_key));
        }
        request_builder = request_builder.header("Content-Type", "application/json");

        if is_streaming {
            // Streaming passthrough
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
            // Non-streaming passthrough
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
        // Conversion mode: convert to Chat Completions format
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

        // Convert Responses API request to Chat Completions format
        let chat_req = convert_responses_to_chat(&req);

        if is_streaming {
            // Need to wrap the response stream to convert back to Responses API format
            match handle_streaming(&state, chat_req.clone(), route, backend_url).await {
                Ok(response) => {
                    // Extract the body and convert from Chat Completions to Responses API format
                    let (parts, body) = response.into_parts();

                    // Generate response ID and created timestamp
                    let response_id = format!("resp_{}", std::process::id());
                    let created_at = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    let model = chat_req.model.clone();
                    // Count prompt tokens for usage field (completion tokens not easily available in stream)
                    let prompt_tokens = count_prompt_tokens(&chat_req) as u32;

                    // Create transformed stream
                    let transformed_stream = convert_chat_sse_to_responses_sse(
                        body,
                        response_id,
                        model,
                        created_at,
                        prompt_tokens,
                    );

                    // Build response with transformed stream
                    let mut response = Response::new(Body::from_stream(transformed_stream));
                    *response.status_mut() = parts.status;
                    *response.headers_mut() = parts.headers;
                    response.into_response()
                }
                Err((status, error)) => (status, error).into_response(),
            }
        } else {
            // Convert Chat Completions response back to Responses API format
            match handle_non_streaming(&state, chat_req, route, backend_url).await {
                Ok(chat_resp) => {
                    let created_at = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs() as i64)
                        .unwrap_or(0);
                    let responses_resp = convert_chat_to_responses(&chat_resp, created_at);
                    (StatusCode::OK, AxumJson(responses_resp)).into_response()
                }
                Err((status, error)) => (status, error).into_response(),
            }
        }
    }
}

/// Convert Chat Completions SSE stream to Responses API SSE stream
fn convert_chat_sse_to_responses_sse(
    body: Body,
    response_id: String,
    model: String,
    created_at: i64,
    prompt_tokens: u32,
) -> impl futures_util::Stream<Item = Result<Bytes, std::io::Error>> {
    let bytes_stream = body.into_data_stream();

    let initial_state = (
        bytes_stream,
        String::new(),
        response_id,
        model,
        created_at,
        prompt_tokens,
        true,                             // is_first_chunk - send response.created first
        false,                            // message_item_added_sent
        false,                            // completed_sent - track if completion events were sent
        std::collections::HashSet::new(), // function_call_indices_sent
    );

    futures_util::stream::unfold(
        initial_state,
        |(
            mut bytes_stream,
            mut buffer,
            response_id,
            model,
            created_at,
            prompt_tokens,
            is_first_chunk,
            mut message_item_added_sent,
            completed_sent,
            mut function_call_indices_sent,
        )| async move {
            if is_first_chunk {
                // Send response.created event first
                let (_, event_data) =
                    create_response_created_event(&response_id, &model, created_at);
                let sse_line = format_responses_sse_event("response.created", &event_data);
                return Some((
                    Ok(Bytes::from(sse_line)),
                    (
                        bytes_stream,
                        buffer,
                        response_id,
                        model,
                        created_at,
                        prompt_tokens,
                        false,
                        message_item_added_sent,
                        completed_sent,
                        function_call_indices_sent,
                    ),
                ));
            }

            loop {
                // Process complete lines from buffer
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[0..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() || !line.starts_with("data: ") {
                        continue;
                    }

                    let data = &line["data: ".len()..];

                    if data.trim() == "[DONE]" {
                        // End of stream - send output_item.done and completed events
                        if !completed_sent {
                            let output_item_id = format!("item_{}", &response_id[0..8]);
                            let (_, done_event) =
                                create_response_output_item_done_event(&output_item_id, created_at);
                            let done_line = format_responses_sse_event(
                                "response.output_item.done",
                                &done_event,
                            );

                            // response.completed event with full response object
                            // Note: For streaming responses, exact token counts are not easily available
                            // here; clients typically rely on response.created for prompt tokens and
                            // sum deltas for output tokens, or use non-streaming mode for accurate usage
                            let completed_event = json!({
                                "type": "response.completed",
                                "created_at": created_at,
                                "response": {
                                    "id": response_id,
                                    "object": "response",
                                    "created_at": created_at,
                                    "model": model,
                                    "status": "completed",
                                    "output": [],
                                    "usage": {
                                        "input_tokens": prompt_tokens,
                                        "output_tokens": 0,
                                        "total_tokens": prompt_tokens
                                    }
                                }
                            });
                            let completed_line =
                                format_responses_sse_event("response.completed", &completed_event);

                            let final_line = format_responses_sse_done();

                            // Combine all final events
                            let combined = format!("{}{}{}", done_line, completed_line, final_line);
                            return Some((
                                Ok(Bytes::from(combined)),
                                (
                                    bytes_stream,
                                    buffer,
                                    response_id,
                                    model,
                                    created_at,
                                    prompt_tokens,
                                    false,
                                    message_item_added_sent,
                                    true,
                                    function_call_indices_sent,
                                ),
                            ));
                        }
                        return None;
                    }

                    // Parse Chat Completions chunk
                    match serde_json::from_str::<OpenAIStreamChunk>(data) {
                        Ok(chunk) => {
                            // Trace: Log raw chunk structure for tool call debugging
                            if !chunk.choices.is_empty() {
                                let choice = &chunk.choices[0];
                                if choice.delta.tool_calls.is_some() {
                                    tracing::trace!(
                                        "Received tool_calls in delta: {:?}",
                                        choice.delta.tool_calls
                                    );
                                }
                                if let Some(ref content) = choice.delta.content
                                    && (content.contains("tool_calls_section")
                                        || content.contains("<|tool_call"))
                                {
                                    tracing::trace!(
                                        "Tool call special tokens found in content: {}",
                                        content
                                    );
                                }
                            }

                            let mut sse_output = String::new();

                            // Process each choice to handle initialization
                            for choice in &chunk.choices {
                                // Check for text content and initialize if needed
                                if let Some(ref content) = choice.delta.content
                                    && !content.is_empty()
                                    && !message_item_added_sent
                                {
                                    let output_item_id = format!("item_{}", &response_id[0..8]);
                                    // output_item.added event for message
                                    let output_item_added = json!({
                                        "type": "response.output_item.added",
                                        "output_index": 0,
                                        "item": {
                                            "id": output_item_id,
                                            "type": "message",
                                            "status": "in_progress",
                                            "role": "assistant",
                                            "content": []
                                        },
                                        "created_at": created_at
                                    });
                                    sse_output.push_str(&format_responses_sse_event(
                                        "response.output_item.added",
                                        &output_item_added,
                                    ));

                                    // content_part.added event
                                    let content_part_added = json!({
                                        "type": "response.content_part.added",
                                        "output_index": 0,
                                        "content_index": 0,
                                        "part": {
                                            "type": "text",
                                            "text": ""
                                        },
                                        "created_at": created_at
                                    });
                                    sse_output.push_str(&format_responses_sse_event(
                                        "response.content_part.added",
                                        &content_part_added,
                                    ));
                                    message_item_added_sent = true;
                                }

                                // Check for tool calls and initialize each function_call if needed
                                if let Some(ref tool_calls) = choice.delta.tool_calls {
                                    for tool_call in tool_calls {
                                        let call_index = tool_call.index.unwrap_or(0);
                                        let output_index = call_index + 1; // text is index 0

                                        if !function_call_indices_sent.contains(&output_index) {
                                            // Initialize function_call output_item
                                            let func_item_id = format!(
                                                "func_{}_{}",
                                                &response_id[0..8],
                                                output_index
                                            );
                                            let func_name = tool_call
                                                .function
                                                .as_ref()
                                                .and_then(|f| f.name.as_ref())
                                                .map(|s| s.as_str())
                                                .unwrap_or("");

                                            let func_item_added = json!({
                                                "type": "response.output_item.added",
                                                "output_index": output_index,
                                                "item": {
                                                    "id": func_item_id,
                                                    "type": "function_call",
                                                    "status": "in_progress",
                                                    "name": func_name,
                                                    "arguments": ""
                                                },
                                                "created_at": created_at
                                            });
                                            sse_output.push_str(&format_responses_sse_event(
                                                "response.output_item.added",
                                                &func_item_added,
                                            ));
                                            function_call_indices_sent.insert(output_index);
                                        }
                                    }
                                }
                            }

                            // Convert to Responses API events
                            let events = convert_chat_stream_chunk_to_responses(
                                &chunk,
                                &response_id,
                                created_at,
                            );
                            for (event_type, event_data) in events {
                                sse_output.push_str(&format_responses_sse_event(
                                    &event_type,
                                    &event_data,
                                ));
                            }

                            if !sse_output.is_empty() {
                                return Some((
                                    Ok(Bytes::from(sse_output)),
                                    (
                                        bytes_stream,
                                        buffer,
                                        response_id,
                                        model,
                                        created_at,
                                        prompt_tokens,
                                        false,
                                        message_item_added_sent,
                                        completed_sent,
                                        function_call_indices_sent,
                                    ),
                                ));
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to parse Chat Completions chunk: {}", e);
                        }
                    }
                }

                // No complete lines - read more data
                match bytes_stream.next().await {
                    Some(Ok(bytes)) => match String::from_utf8(bytes.to_vec()) {
                        Ok(s) => buffer.push_str(&s),
                        Err(_) => tracing::warn!("Received invalid UTF-8 in stream"),
                    },
                    Some(Err(e)) => {
                        tracing::error!("Stream error: {}", e);
                        return Some((
                            Err(std::io::Error::other(e)),
                            (
                                bytes_stream,
                                buffer,
                                response_id,
                                model,
                                created_at,
                                prompt_tokens,
                                false,
                                message_item_added_sent,
                                completed_sent,
                                function_call_indices_sent,
                            ),
                        ));
                    }
                    None => {
                        // Stream ended
                        if !completed_sent {
                            // Send completion events
                            let output_item_id = format!("item_{}", &response_id[0..8]);
                            let (_, done_event) =
                                create_response_output_item_done_event(&output_item_id, created_at);
                            let done_line = format_responses_sse_event(
                                "response.output_item.done",
                                &done_event,
                            );

                            // response.completed event with full response object
                            // Note: For streaming responses, exact token counts are not easily available
                            // here; clients typically rely on response.created for prompt tokens and
                            // sum deltas for output tokens, or use non-streaming mode for accurate usage
                            let completed_event = json!({
                                "type": "response.completed",
                                "created_at": created_at,
                                "response": {
                                    "id": response_id,
                                    "object": "response",
                                    "created_at": created_at,
                                    "model": model,
                                    "status": "completed",
                                    "output": [],
                                    "usage": {
                                        "input_tokens": prompt_tokens,
                                        "output_tokens": 0,
                                        "total_tokens": prompt_tokens
                                    }
                                }
                            });
                            let completed_line =
                                format_responses_sse_event("response.completed", &completed_event);

                            let final_line = format_responses_sse_done();

                            let combined = format!("{}{}{}", done_line, completed_line, final_line);
                            return Some((
                                Ok(Bytes::from(combined)),
                                (
                                    bytes_stream,
                                    buffer,
                                    response_id,
                                    model,
                                    created_at,
                                    prompt_tokens,
                                    false,
                                    message_item_added_sent,
                                    true,
                                    function_call_indices_sent,
                                ),
                            ));
                        }
                        return None;
                    }
                }
            }
        },
    )
}

/// Build upstream models URL from route config
fn build_models_url(route: &RouteConfig) -> Option<String> {
    match route.provider_type {
        ProviderType::OpenAi => route
            .base_url
            .as_ref()
            .map(|base_url| format!("{}/v1/models", base_url.trim_end_matches('/'))),
        ProviderType::OpenAiCompatible => {
            // Try to extract base_url from the full chat completions URL
            route.url.as_ref().and_then(|url| {
                url.find("/v1/chat/completions")
                    .map(|idx| format!("{}/v1/models", &url[..idx]))
                    .or_else(|| {
                        route
                            .base_url
                            .as_ref()
                            .map(|base_url| format!("{}/v1/models", base_url.trim_end_matches('/')))
                    })
            })
        }
        ProviderType::Ollama | ProviderType::Anthropic | ProviderType::Gemini => None,
        ProviderType::Moonlight => None, // Moonlight doesn't support model listing endpoint
    }
}

/// Fetch model information from upstream provider
async fn fetch_upstream_model_info(
    client: &reqwest::Client,
    route: &RouteConfig,
    created: u64,
) -> OpenAIModel {
    // Try to fetch from upstream if supported
    if let Some(models_url) = build_models_url(route) {
        let mut request = client.get(&models_url);
        if let Some(api_key) = &route.api_key {
            request = request.header("Authorization", format!("Bearer {}", api_key));
        }
        match request
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(upstream_response) = resp.json::<OpenAIModelsListResponse>().await {
                    // Find matching model in upstream response
                    let upstream_model_name = route.upstream_model();
                    if let Some(upstream_model) = upstream_response
                        .data
                        .iter()
                        .find(|m| m.id == upstream_model_name || m.id == route.model_name)
                    {
                        return OpenAIModel {
                            id: route.model_name.clone(),
                            object: "model".to_string(),
                            created: upstream_model.created,
                            owned_by: upstream_model.owned_by.clone(),
                            root: upstream_model.root.clone(),
                            parent: upstream_model.parent.clone(),
                            max_model_len: upstream_model.max_model_len,
                        };
                    }
                }
            }
            _ => {}
        }
    }

    // Fallback: return basic model info with defaults
    let owned_by = match route.provider_type {
        ProviderType::OpenAi => "openai",
        ProviderType::Anthropic => "anthropic",
        ProviderType::Ollama => "ollama",
        ProviderType::Gemini => "google",
        ProviderType::Moonlight => "moonlight",
        ProviderType::OpenAiCompatible => "sglang",
    };

    OpenAIModel {
        id: route.model_name.clone(),
        object: "model".to_string(),
        created,
        owned_by: owned_by.to_string(),
        root: Some(route.upstream_model().to_string()),
        parent: None,
        max_model_len: None,
    }
}

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
    // Load and validate new configuration
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

    // Get current config to compare fields
    let current_config = state.config.load();

    // Check if stats enabled status changed and rebuild StatsWriter if needed
    if new_config.statistics.enabled != current_config.statistics.enabled {
        if new_config.statistics.enabled {
            // Enable statistics
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
            // Disable statistics
            state.stats_writer.store(Arc::new(None));
            tracing::info!("Statistics disabled during config reload");
        }
    }

    // Collect warnings about fields that require restart
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

    // Update the configuration atomically
    state.config.store(Arc::new(new_config));
    tracing::info!("Configuration reloaded successfully");

    // Return success response with warnings
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

    // Build routes with provider_type only (no api keys exposed)
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
