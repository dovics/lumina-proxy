//! Responses API conversion utilities

use crate::config::{ProviderType, RouteConfig};
use crate::convert::{
    convert_chat_stream_chunk_to_responses, create_response_created_event,
    create_response_output_item_done_event, format_responses_sse_done, format_responses_sse_event,
};
use crate::types::*;
use axum::body::Body;
use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::json;

// =============================================================================
// Responses API SSE Conversion
// =============================================================================

/// Convert Chat Completions SSE stream to Responses API SSE stream
pub fn convert_chat_sse_to_responses_sse(
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
        true,
        false,
        false,
        std::collections::HashSet::new(),
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
            mut completed_sent,
            mut function_call_indices_sent,
        )| async move {
            if is_first_chunk {
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
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[0..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() || !line.starts_with("data: ") {
                        continue;
                    }

                    let data = &line["data: ".len()..];

                    if data.trim() == "[DONE]" {
                        if !completed_sent {
                            let output_item_id = format!("item_{}", &response_id[0..8]);
                            let (_, done_event) =
                                create_response_output_item_done_event(&output_item_id, created_at);
                            let done_line = format_responses_sse_event(
                                "response.output_item.done",
                                &done_event,
                            );

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

                    match serde_json::from_str::<OpenAIStreamChunk>(data) {
                        Ok(chunk) => {
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
                            let mut debug_tool_call_events = Vec::new();

                            for choice in &chunk.choices {
                                if let Some(ref content) = choice.delta.content
                                    && !content.is_empty()
                                    && !message_item_added_sent
                                {
                                    let output_item_id = format!("item_{}", &response_id[0..8]);
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

                                if let Some(ref tool_calls) = choice.delta.tool_calls {
                                    for tool_call in tool_calls {
                                        let call_index = tool_call.index.unwrap_or(0);
                                        let output_index = call_index + 1;

                                        if !function_call_indices_sent.contains(&output_index) {
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

                                            debug_tool_call_events.push(format!(
                                                "func_item_added: func_item_id={}, func_name={}",
                                                func_item_id, func_name
                                            ));
                                        }

                                        if let Some(ref function) = tool_call.function {
                                            if let Some(ref name) = function.name {
                                                debug_tool_call_events.push(format!(
                                                    "function_call event: name={}, call_id={}",
                                                    name,
                                                    tool_call
                                                        .id
                                                        .as_ref()
                                                        .unwrap_or(&"<none>".to_string())
                                                ));
                                            }
                                            if let Some(ref args) = function.arguments {
                                                debug_tool_call_events.push(format!(
                                                    "function_call_arguments.delta event: args={}",
                                                    if args.len() > 100 {
                                                        format!("{}...<truncated>", &args[..100])
                                                    } else {
                                                        args.clone()
                                                    }
                                                ));
                                            }
                                        }
                                    }
                                }
                            }

                            if !debug_tool_call_events.is_empty() {
                                tracing::debug!(
                                    "Responses API conversion: chunk_id={}, output_events={:?}",
                                    chunk.id,
                                    debug_tool_call_events
                                );
                            }

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

                            // Check if this is a final chunk with finish_reason == "tool_calls"
                            // In this case, we need to send the completion events
                            let is_tool_calls_final = chunk
                                .choices
                                .iter()
                                .any(|c| c.finish_reason.as_deref() == Some("tool_calls"));

                            if is_tool_calls_final && !completed_sent {
                                tracing::debug!(
                                    "Responses API: detected tool_calls finish, sending completion events"
                                );

                                // Send output_item.done for function call items
                                for output_index in &function_call_indices_sent {
                                    let func_item_id =
                                        format!("func_{}_{}", &response_id[0..8], output_index);
                                    let done_event = json!({
                                        "type": "response.output_item.done",
                                        "output_index": output_index,
                                        "item": {
                                            "id": func_item_id,
                                            "type": "function_call",
                                            "status": "completed",
                                            "name": "",
                                            "arguments": ""
                                        },
                                        "created_at": created_at
                                    });
                                    let done_line = format_responses_sse_event(
                                        "response.output_item.done",
                                        &done_event,
                                    );
                                    sse_output.push_str(&done_line);
                                }

                                // Send response.completed
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
                                let completed_line = format_responses_sse_event(
                                    "response.completed",
                                    &completed_event,
                                );
                                sse_output.push_str(&completed_line);

                                // Mark as completed so we don't send these again
                                completed_sent = true;
                            }

                            if !sse_output.is_empty() {
                                tracing::trace!(
                                    "Responses API SSE output: {} bytes, events: {}",
                                    sse_output.len(),
                                    sse_output.lines().count()
                                );
                                for line in sse_output.lines().take(10) {
                                    tracing::trace!("SSE line: {}", line);
                                }
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
                        if !completed_sent {
                            let output_item_id = format!("item_{}", &response_id[0..8]);
                            let (_, done_event) =
                                create_response_output_item_done_event(&output_item_id, created_at);
                            let done_line = format_responses_sse_event(
                                "response.output_item.done",
                                &done_event,
                            );

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

// =============================================================================
// Model Info Utilities
// =============================================================================

/// Build upstream models URL from route config
pub fn build_models_url(route: &RouteConfig) -> Option<String> {
    match route.provider_type {
        ProviderType::OpenAi => route
            .base_url
            .as_ref()
            .map(|base_url| format!("{}/v1/models", base_url.trim_end_matches('/'))),
        ProviderType::OpenAiCompatible => route.url.as_ref().and_then(|url| {
            url.find("/v1/chat/completions")
                .map(|idx| format!("{}/v1/models", &url[..idx]))
                .or_else(|| {
                    route
                        .base_url
                        .as_ref()
                        .map(|base_url| format!("{}/v1/models", base_url.trim_end_matches('/')))
                })
        }),
        ProviderType::Ollama | ProviderType::Anthropic | ProviderType::Gemini => None,
    }
}

/// Fetch model information from upstream provider
pub async fn fetch_upstream_model_info(
    client: &reqwest::Client,
    route: &RouteConfig,
    created: u64,
) -> OpenAIModel {
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
