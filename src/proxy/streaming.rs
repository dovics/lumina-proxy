//! Streaming request handling

use super::state::ProxyState;
use super::tools::aggregate_tool_calls;
use crate::config::{ProviderType, RouteConfig};
use crate::convert::*;
use crate::stats::RequestMetrics;
use crate::token_counter::*;
use crate::types::*;
use axum::{
    Json,
    body::Body,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Instant;

// =============================================================================
// Streaming Chunk Result
// =============================================================================

/// Result type for streaming chunk parsing
enum StreamChunkResult {
    /// A single streaming chunk
    Single(OpenAIStreamChunk),
    /// Parse error
    Error(String),
}

// =============================================================================
// Streaming Request Handling
// =============================================================================

/// Handle a streaming chat completion request
pub async fn handle_streaming(
    state: &ProxyState,
    req: OpenAIChatRequest,
    route: &RouteConfig,
    backend_url: String,
) -> Result<axum::response::Response<Body>, (StatusCode, Json<serde_json::Value>)> {
    let model = req.model.clone();
    let start_time = std::time::Instant::now();
    let first_bytes_time = Arc::new(Mutex::new(None::<Instant>));

    let mut streaming_req = req.clone();
    streaming_req.stream = Some(true);
    streaming_req.model = route.upstream_model().to_string();

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
    };

    let body = match body {
        Ok(b) => b,
        Err(e) => return Err(e),
    };

    // Debug logging for GLM/OpenAI-compatible backends
    if matches!(route.provider_type, ProviderType::OpenAiCompatible)
        && let Ok(body_str) = String::from_utf8(body.clone())
    {
        tracing::debug!(
            "OpenAI-compatible request body: {}",
            if body_str.len() > 2000 { format!("{}...", &body_str[..2000]) } else { body_str }
        );
    }

    let mut request_builder = state.client.post(backend_url.clone());
    if let Some(api_key) = &route.api_key {
        request_builder = request_builder.header("Authorization", format!("Bearer {}", api_key));
    }
    request_builder = request_builder.header("Content-Type", "application/json");
    request_builder = request_builder.header("Accept", "text/event-stream");

    let response = request_builder.body(body).send().await.map_err(|e| {
        tracing::error!(
            backend_url = %backend_url,
            error = %e,
            "Backend connection failed"
        );
        (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Backend request failed: {}", e) })),
        )
    })?;

    let status = response.status();
    tracing::debug!(
        backend_url = %backend_url,
        response_status = %status,
        "backend response status"
    );

    if !status.is_success() {
        let error_text = response.text().await.unwrap_or_default();
        tracing::error!(
            backend_url = %backend_url,
            status = %status,
            error_response = %error_text,
            content_length = error_text.len(),
            "Backend returned error"
        );
        return Err((
            status,
            Json(json!({ "error": format!("Backend returned error: {}", error_text) })),
        ));
    }

    // Check Content-Type to detect error pages (e.g., Kimi returning HTML error pages with 200 status)
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();

    if !content_type.contains("text/event-stream") && !content_type.contains("application/json") {
        let body_text = response.text().await.unwrap_or_default();
        tracing::error!(
            backend_url = %backend_url,
            content_type = %content_type,
            body_preview = %if body_text.len() > 500 { format!("{}...", &body_text[..500]) } else { body_text.clone() },
            "Backend returned non-SSE response (possibly error page)"
        );
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": format!("Backend returned unexpected content type: {}", content_type) })),
        ));
    }

    let bytes_stream = response.bytes_stream();

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

    let pinned_bytes_stream = Box::pin(bytes_stream);
    let initial_state = (
        pinned_bytes_stream,
        token_counter_clone,
        response_id.clone(),
        created,
        model_clone,
        String::new(),
        first_bytes_time.clone(),
        true,
    );

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

            loop {
                let mut yielded_chunks: Vec<Result<Bytes, std::io::Error>> = Vec::new();
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[0..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() || !line.starts_with("data: ") {
                        continue;
                    }

                    let data = &line["data: ".len()..];

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

                    if data.trim() == "[DONE]" {
                        continue;
                    }

                    let openai_chunk_result: StreamChunkResult = match provider_type {
                        ProviderType::Ollama => {
                            match serde_json::from_str::<OllamaStreamChunk>(data) {
                                Ok(ollama_chunk) => {
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
                            match serde_json::from_str::<GeminiStreamChunk>(data) {
                                Ok(gemini_chunk) => {
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
                            let raw_value: serde_json::Value = match serde_json::from_str(data) {
                                Ok(v) => v,
                                Err(_) => serde_json::Value::Null,
                            };

                            match serde_json::from_str::<OpenAIStreamChunk>(data) {
                                Ok(mut openai_chunk) => {
                                    let mut has_content = false;
                                    for choice in &openai_chunk.choices {
                                        if let Some(content) = &choice.delta.content
                                            && !content.is_empty()
                                        {
                                            counter.add_delta(content);
                                            has_content = true;
                                        }
                                        if let Some(reasoning) = &choice.delta.reasoning
                                            && !reasoning.is_empty()
                                        {
                                            counter.add_delta(reasoning);
                                            has_content = true;
                                        }
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

                                    for choice in &mut openai_chunk.choices {
                                        if let Some(ref tool_calls) = choice.delta.tool_calls {
                                            let aggregated = aggregate_tool_calls(tool_calls);
                                            if !aggregated.is_empty() {
                                                choice.delta.tool_calls = Some(aggregated);
                                            }
                                        }
                                    }

                                    if !has_content {
                                        tracing::trace!(
                                            "Chunk parsed but no countable content: {}",
                                            serde_json::to_string(&raw_value)
                                                .unwrap_or_else(|_| data.to_string())
                                        );
                                    }

                                    openai_chunk.id = id.clone();
                                    openai_chunk.model = model.clone();
                                    StreamChunkResult::Single(openai_chunk)
                                }
                                Err(e) => {
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

                    };

                    match openai_chunk_result {
                        StreamChunkResult::Single(mut chunk) => {
                            if is_first_chunk {
                                chunk.usage = Some(OpenAIUsage {
                                    prompt_tokens: prompt_tokens_stream as u32,
                                    completion_tokens: 0,
                                    total_tokens: prompt_tokens_stream as u32,
                                });
                            }

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

                            let sse_line =
                                format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap());
                            yielded_chunks.push(Ok(Bytes::from(sse_line)));
                        }
                        StreamChunkResult::Error(e) => {
                            tracing::warn!("{}", e);
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

                if !yielded_chunks.is_empty() {
                    // Concatenate ALL available chunks to avoid per-chunk latency
                    // Without this, only 1 chunk is sent per upstream network read,
                    // causing massive latency when upstream sends many chunks at once
                    let mut all_bytes = Vec::new();
                    for bytes in yielded_chunks.into_iter().flatten() {
                        all_bytes.extend_from_slice(&bytes);
                    }
                    return Some((
                        Ok(Bytes::from(all_bytes)),
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

                        {
                            let mut first_time = first_bytes_time.lock().await;
                            if first_time.is_none() {
                                *first_time = Some(Instant::now());
                            }
                        }

                        match String::from_utf8(bytes.to_vec()) {
                            Ok(s) => buffer.push_str(&s),
                            Err(_) => {
                                tracing::warn!("Received invalid UTF-8 in stream");
                            }
                        }
                    }
                    None => {
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
            0,
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

            use futures_util::StreamExt;
            match stream.next().await {
                Some(item) => Some((
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
                )),
                None => {
                    let completion_tokens = token_counter_clone.total();
                    let total_tokens = prompt_tokens + completion_tokens;
                    let duration_ms = start_time.elapsed().as_millis() as u64;

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
