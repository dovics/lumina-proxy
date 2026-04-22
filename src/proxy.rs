//! Core proxy implementation - handles request routing, conversion, and streaming

use arc_swap::ArcSwap;
use axum::{
    body::{Bytes, Body},
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use axum::response::Json as AxumJson;
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use chrono::Utc;

use crate::config::{Config, ProviderType, RouteConfig};
use crate::types::*;
use crate::convert::*;
use crate::token_counter::*;
use crate::stats::StatsWriter;
use crate::stats::TokenStats;

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
    match route.provider_type {
        ProviderType::Ollama => {
            let base_url = route.base_url
                .as_ref()
                .ok_or_else(|| ProxyError::ConfigError(
                    format!("Ollama route for model '{}' missing base_url", model)
                ))?;
            Ok(format!("{}/api/chat", base_url.trim_end_matches('/')))
        }

        ProviderType::Anthropic => {
            let base_url = route.base_url
                .as_ref()
                .ok_or_else(|| ProxyError::ConfigError(
                    format!("Anthropic route for model '{}' missing base_url", model)
                ))?;
            Ok(format!("{}/v1/messages", base_url.trim_end_matches('/')))
        }

        ProviderType::Gemini => {
            let base_url = route.base_url
                .as_ref()
                .ok_or_else(|| ProxyError::ConfigError(
                    format!("Gemini route for model '{}' missing base_url", model)
                ))?;
            Ok(format!(
                "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
                base_url.trim_end_matches('/'),
                route.upstream_model()
            ))
        }

        ProviderType::OpenAi => {
            if let Some(url) = &route.url {
                Ok(url.clone())
            } else if let Some(base_url) = &route.base_url {
                Ok(format!("{}/v1/chat/completions", base_url.trim_end_matches('/')))
            } else {
                Err(ProxyError::ConfigError(
                    format!("OpenAI route for model '{}' missing either url or base_url", model)
                ))
            }
        }

        ProviderType::OpenAiCompatible => {
            route.url
                .clone()
                .ok_or_else(|| ProxyError::ConfigError(
                    format!("OpenAI-compatible route for model '{}' missing url", model)
                ))
        }
    }
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

    // Add authorization
    request_builder = request_builder.header("Authorization", format!("Bearer {}", route.api_key));
    // Set content type to JSON since we're sending JSON body
    request_builder = request_builder.header("Content-Type", "application/json");

    // Convert request body based on provider
    let body = match route.provider_type {
        ProviderType::Ollama => serde_json::to_vec(&convert_openai_to_ollama(&outgoing_req))
            .map_err(|e| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize Ollama request: {}", e) }))
            )),

        ProviderType::Anthropic => serde_json::to_vec(&convert_openai_to_anthropic(&outgoing_req))
            .map_err(|e| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize Anthropic request: {}", e) }))
            )),

        ProviderType::Gemini => serde_json::to_vec(&convert_openai_to_gemini(&outgoing_req))
            .map_err(|e| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize Gemini request: {}", e) }))
            )),

        ProviderType::OpenAi | ProviderType::OpenAiCompatible => serde_json::to_vec(&outgoing_req)
            .map_err(|e| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize OpenAI request: {}", e) }))
            )),
    };

    let body = match body {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    // Send the request
    let response = request_builder
        .body(body)
        .send()
        .await
        .map_err(|e| (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Backend request failed: {}", e) }))
        ))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err((
            status,
            Json(json!({ "error": format!("Backend returned error: {}", error_text) }))
        ));
    }

    // Get completion tokens based on provider
    let (openai_resp, completion_tokens) = match route.provider_type {
        ProviderType::Ollama => {
            let ollama_resp: OllamaChatResponse = response.json()
                .await
                .map_err(|e| (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Failed to parse Ollama response: {}", e) }))
                ))?;
            let resp = convert_ollama_to_openai(&ollama_resp, &model);
            let tokens = resp.usage.completion_tokens as usize;
            (resp, tokens)
        }

        ProviderType::Anthropic => {
            let anthropic_resp: AnthropicChatResponse = response.json()
                .await
                .map_err(|e| (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Failed to parse Anthropic response: {}", e) }))
                ))?;
            let resp = convert_anthropic_to_openai(&anthropic_resp, &model);
            let tokens = resp.usage.completion_tokens as usize;
            (resp, tokens)
        }

        ProviderType::Gemini => {
            let gemini_resp: GeminiChatResponse = response.json()
                .await
                .map_err(|e| (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Failed to parse Gemini response: {}", e) }))
                ))?;
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let resp = convert_gemini_to_openai(&gemini_resp, &model, created);
            let tokens = resp.usage.completion_tokens as usize;
            (resp, tokens)
        }

        ProviderType::OpenAi | ProviderType::OpenAiCompatible => {
            let mut openai_resp: OpenAIChatResponse = response.json()
                .await
                .map_err(|e| (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({ "error": format!("Failed to parse OpenAI response: {}", e) }))
                ))?;
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
        let stats = TokenStats {
            timestamp: Utc::now(),
            model: model.clone(),
            provider: format!("{:?}", route.provider_type).to_lowercase(),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            duration_ms,
            status: "success".to_string(),
            error_message: None,
        };
        if let Err(e) = stats_writer.write_stat(stats).await {
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

    // Convert request body based on provider - use upstream_model for outgoing request
    let mut streaming_req = req.clone();
    streaming_req.stream = Some(true);
    streaming_req.model = route.upstream_model().to_string();

    // Count prompt tokens synchronously (fast enough) - use upstream model for correct tokenizer selection
    let prompt_tokens = count_prompt_tokens(&streaming_req);

    let body = match route.provider_type {
        ProviderType::Ollama => serde_json::to_vec(&convert_openai_to_ollama(&streaming_req))
            .map_err(|e| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize Ollama request: {}", e) }))
            )),

        ProviderType::Anthropic => serde_json::to_vec(&convert_openai_to_anthropic(&streaming_req))
            .map_err(|e| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize Anthropic request: {}", e) }))
            )),

        ProviderType::Gemini => serde_json::to_vec(&convert_openai_to_gemini(&streaming_req))
            .map_err(|e| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize Gemini request: {}", e) }))
            )),

        ProviderType::OpenAi | ProviderType::OpenAiCompatible => serde_json::to_vec(&streaming_req)
            .map_err(|e| (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Failed to serialize OpenAI request: {}", e) }))
            )),
    };

    let body = match body {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    // Build and send the request
    let mut request_builder = state.client.post(backend_url.clone());
    request_builder = request_builder.header("Authorization", format!("Bearer {}", route.api_key));
    // Set content type to JSON since we're sending JSON body
    request_builder = request_builder.header("Content-Type", "application/json");

    // Some providers require specific headers for SSE
    request_builder = request_builder.header("Accept", "text/event-stream");

    let response = request_builder
        .body(body)
        .send()
        .await
        .map_err(|e| (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Backend request failed: {}", e) }))
        ))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err((
            status,
            Json(json!({ "error": format!("Backend returned error: {}", error_text) }))
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
    );

    let transformed_stream = futures_util::stream::unfold(
        initial_state,
        move |(mut bytes_stream, counter, id, created, model, mut buffer)| async move {
            let provider_type = provider_type;

            // Get next chunk from backend
            match bytes_stream.next().await {
                Some(bytes_result) => {
                    let bytes = match bytes_result {
                        Ok(b) => b,
                        Err(e) => {
                            tracing::error!("Stream error: {}", e);
                            let error_chunk: Result<Bytes, reqwest::Error> = Ok(Bytes::from(format!(
                                "data: {{\"error\": \"{}\"}}\n\n",
                                e
                            )));
                            return Some((
                                error_chunk,
                                (bytes_stream, counter, id, created, model, buffer)
                            ));
                        }
                    };

                    // Convert bytes to string and append to buffer
                    match String::from_utf8(bytes.to_vec()) {
                        Ok(s) => buffer.push_str(&s),
                        Err(_) => {
                            tracing::warn!("Received invalid UTF-8 in stream");
                        }
                    }
                }
                None => {
                    // End of stream - if buffer is empty, we're done
                    if buffer.is_empty() {
                        return None;
                    }
                }
            }

            // Process lines from buffer (SSE uses newline delimiters)
            let mut yielded_chunks = Vec::new();
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[0..pos].trim().to_string();
                buffer = buffer[pos+1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line["data: ".len()..];

                // Skip [DONE] message for now, we'll add it at the end
                if data.trim() == "[DONE]" {
                    continue;
                }

                // Parse and convert chunk based on provider
                let openai_chunk_result: Result<OpenAIStreamChunk, String> = match provider_type {
                    ProviderType::Ollama => {
                        match serde_json::from_str::<OllamaStreamChunk>(data) {
                            Ok(ollama_chunk) => {
                                // Count tokens incrementally
                                if let Some(delta) = &ollama_chunk.delta {
                                    counter.add_delta(&delta.content);
                                }
                                Ok(convert_ollama_stream_chunk_to_openai(
                                    &ollama_chunk,
                                    &id,
                                    created,
                                    &model
                                ))
                            }
                            Err(e) => {
                                Err(format!("Failed to parse Ollama stream chunk: {}", e))
                            }
                        }
                    }

                    ProviderType::Anthropic => {
                        match serde_json::from_str::<AnthropicStreamChunk>(data) {
                            Ok(anthropic_chunk) => {
                                if let Some(delta) = &anthropic_chunk.delta && let Some(text) = &delta.text {
                                    counter.add_delta(text);
                                }
                                Ok(convert_anthropic_stream_chunk_to_openai(
                                    &anthropic_chunk,
                                    created,
                                    &model
                                ))
                            }
                            Err(e) => {
                                Err(format!("Failed to parse Anthropic stream chunk: {}", e))
                            }
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
                                Ok(convert_gemini_stream_chunk_to_openai(
                                    &gemini_chunk,
                                    &id,
                                    created,
                                    &model
                                ))
                            }
                            Err(e) => {
                                Err(format!("Failed to parse Gemini stream chunk: {}", e))
                            }
                        }
                    }

                    ProviderType::OpenAi | ProviderType::OpenAiCompatible => {
                        match serde_json::from_str::<OpenAIStreamChunk>(data) {
                            Ok(mut openai_chunk) => {
                                // Count tokens from all choices
                                for choice in &openai_chunk.choices {
                                    if let Some(delta) = &choice.delta.content {
                                        counter.add_delta(delta);
                                    }
                                }
                                // Ensure the response ID is consistent
                                openai_chunk.id = id.clone();
                                // Override model name to client-requested name
                                openai_chunk.model = model.clone();
                                Ok(openai_chunk)
                            }
                            Err(e) => {
                                Err(format!("Failed to parse OpenAI stream chunk: {}", e))
                            }
                        }
                    }
                };

                match openai_chunk_result {
                    Ok(chunk) => {
                        // Format as SSE
                        let sse_line = format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                        yielded_chunks.push(Ok(Bytes::from(sse_line)));
                    }
                    Err(e) => {
                        tracing::warn!("{}", e);
                    }
                }
            }

            if !yielded_chunks.is_empty() {
                // Return the first chunk and keep remaining for next iteration
                let next_chunk = yielded_chunks.remove(0);
                // Any extra chunks get added back to the buffer by converting them
                // This is a bit of a hack but it works
                for bytes in yielded_chunks.into_iter().flatten() {
                    // Convert SSE format back to data line for reprocessing
                    let s = String::from_utf8_lossy(bytes.as_ref());
                    // The SSE is already "data: ...\n\n" so just add it to buffer
                    buffer = s.to_string() + &buffer;
                }
                Some((next_chunk, (bytes_stream, counter, id, created, model, buffer)))
            } else if buffer.is_empty() {
                // No chunks yielded and we're at end of stream
                None
            } else {
                // No chunks in this iteration but we still have buffer content - it means no complete lines yet
                // Just keep waiting for more data, continue the loop by returning None (end of iteration)
                None
            }
        }
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
            false
        ),
        |(mut stream, stats_writer, model, provider_type, prompt_tokens, start_time, token_counter_clone, done)| async move {
            if done {
                return None;
            }

            // Use poll_next with pinned projection to avoid Unpin issue
            use futures_util::StreamExt;
            match stream.next().await {
                Some(item) => Some((item, (stream, stats_writer, model, provider_type, prompt_tokens, start_time, token_counter_clone, false))),
                None => {
                    // Stream is complete, write stats
                    let completion_tokens = token_counter_clone.total();
                    let total_tokens = prompt_tokens + completion_tokens;
                    let duration_ms = start_time.elapsed().as_millis() as u64;

                    tracing::info!(
                        model = %model,
                        provider = %format!("{:?}", provider_type),
                        prompt_tokens = %prompt_tokens,
                        completion_tokens = %completion_tokens,
                        total_tokens = %total_tokens,
                        duration_ms = %duration_ms,
                        "Completed streaming request"
                    );

                    if let Some(stats_writer) = stats_writer.as_ref() {
                        let stats = TokenStats {
                            timestamp: Utc::now(),
                            model: model.clone(),
                            provider: format!("{:?}", provider_type).to_lowercase(),
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            duration_ms,
                            status: "success".to_string(),
                            error_message: None,
                        };
                        if let Err(e) = stats_writer.write_stat(stats).await {
                            tracing::warn!("Failed to write statistics: {}", e);
                        }
                    }

                    // Send final [DONE]
                    Some((
                        Ok(Bytes::from("data: [DONE]\n\n")),
                        (stream, stats_writer, model, provider_type, prompt_tokens, start_time, token_counter_clone, true)
                    ))
                }
            }
        }
    );

    let body = Body::from_stream(final_stream);

    let mut headers = HeaderMap::new();
    headers.insert(
        "Content-Type",
        "text/event-stream".parse().unwrap()
    );
    headers.insert(
        "Cache-Control",
        "no-cache".parse().unwrap()
    );
    headers.insert(
        "Connection",
        "keep-alive".parse().unwrap()
    );

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
            Json(json!({ "error": "Request too large, maximum 100MB allowed" }))
        ).into_response();
    }
    // Parse OpenAI request from body
    let req: OpenAIChatRequest = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("Invalid request body: {}", e) }))
            ).into_response();
        }
    };

    // Find backend route for requested model
    let config = state.config.load();
    let model = req.model.clone();
    let Some(route) = config.find_backend_for_model(&model) else {
        return (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("No backend route configured for model: {}", model) }))
        ).into_response();
    };

    // Build backend URL
    let backend_url = match build_backend_url(route, &model) {
        Ok(url) => url,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": e.to_string() }))
            ).into_response();
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

/// Build upstream models URL from route config
fn build_models_url(route: &RouteConfig) -> Option<String> {
    match route.provider_type {
        ProviderType::OpenAi => {
            route.base_url
                .as_ref()
                .map(|base_url| format!("{}/v1/models", base_url.trim_end_matches('/')))
        }
        ProviderType::OpenAiCompatible => {
            // Try to extract base_url from the full chat completions URL
            route.url.as_ref().and_then(|url| {
                url.find("/v1/chat/completions")
                    .map(|idx| format!("{}/v1/models", &url[..idx]))
                    .or_else(|| route.base_url
                        .as_ref()
                        .map(|base_url| format!("{}/v1/models", base_url.trim_end_matches('/')))
                    )
            })
        }
        ProviderType::Ollama | ProviderType::Anthropic | ProviderType::Gemini => None,
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
        match client
            .get(&models_url)
            .header("Authorization", format!("Bearer {}", route.api_key))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(upstream_response) = resp.json::<OpenAIModelsListResponse>().await {
                    // Find matching model in upstream response
                    let upstream_model_name = route.upstream_model();
                    if let Some(upstream_model) = upstream_response.data
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
pub async fn models_handler(
    State(state): State<Arc<ProxyState>>,
) -> impl IntoResponse {
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
pub async fn reload_config_handler(
    State(state): State<Arc<ProxyState>>,
) -> impl IntoResponse {
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
                }))
            ).into_response();
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
        warnings.push("Logging configuration changes require server restart to take effect".to_string());
    }
    if new_config.statistics.file_path != current_config.statistics.file_path
        || new_config.statistics.buffer_seconds != current_config.statistics.buffer_seconds
    {
        warnings.push("Statistics file/buffer configuration changes require server restart to take effect".to_string());
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
        }))
    ).into_response()
}

/// Handler for GET /v1/admin/config - returns current configuration (sensitive fields masked)
pub async fn get_config_handler(
    State(state): State<Arc<ProxyState>>,
) -> impl IntoResponse {
    let config = state.config.load();

    // Build routes with provider_type only (no api keys exposed)
    let routes: Vec<_> = config.routes.iter().map(|route| {
        json!({
            "model_name": route.model_name,
            "provider_type": format!("{:?}", route.provider_type).to_lowercase(),
            "enabled": route.enabled
        })
    }).collect();

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
