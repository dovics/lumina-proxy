//! Format conversion functions between OpenAI standard and provider-native formats

use crate::types::*;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// OpenAI → Ollama Conversion
// =============================================================================

/// Convert an OpenAI-format chat request to Ollama-native format
pub fn convert_openai_to_ollama(req: &OpenAIChatRequest) -> OllamaChatRequest {
    let messages = req
        .messages
        .iter()
        .map(|m| {
            let content = m.content.clone().unwrap_or_default();
            OllamaMessage {
                role: m.role.clone(),
                content,
            }
        })
        .collect();

    // Convert OpenAI tools to Ollama tools format
    let tools = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter(|t| t.function.is_some())
            .map(|t| OllamaTool {
                type_: "function".to_string(),
                function: OllamaToolFunction {
                    name: t.function.as_ref().unwrap().name.clone(),
                    description: t.function.as_ref().unwrap().description.clone(),
                    parameters: t.function.as_ref().unwrap().parameters.clone(),
                },
            })
            .collect()
    });

    OllamaChatRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        num_predict: req.max_tokens,
        stream: req.stream,
        stop: req.stop.clone(),
        tools,
    }
}

/// Convert an Ollama-native non-streaming response to OpenAI-format
pub fn convert_ollama_to_openai(resp: &OllamaChatResponse, model: &str) -> OpenAIChatResponse {
    let now = current_timestamp();

    OpenAIChatResponse {
        id: format!("ollama-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: model.to_string(),
        choices: vec![OpenAIChoice {
            index: 0,
            message: Some(OpenAIMessage {
                role: resp.message.role.clone(),
                content: Some(resp.message.content.clone()),
                ..Default::default()
            }),
            delta: None,
            finish_reason: if resp.done {
                Some("stop".to_string())
            } else {
                None
            },
        }],
        usage: OpenAIUsage {
            prompt_tokens: resp.prompt_eval_count.unwrap_or(0),
            completion_tokens: resp.eval_count.unwrap_or(0),
            total_tokens: resp.prompt_eval_count.unwrap_or(0) + resp.eval_count.unwrap_or(0),
        },
    }
}

/// Convert an Ollama streaming chunk to OpenAI-format
pub fn convert_ollama_stream_chunk_to_openai(
    chunk: &OllamaStreamChunk,
    id: &str,
    created: u64,
    model: &str,
) -> OpenAIStreamChunk {
    let content = chunk.delta.as_ref().map(|d| d.content.clone());

    OpenAIStreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                role: if chunk.message.is_some() {
                    Some("assistant".to_string())
                } else {
                    None
                },
                content,
                ..Default::default()
            },
            finish_reason: if chunk.done {
                Some("stop".to_string())
            } else {
                None
            },
        }],
        usage: None,
    }
}

// =============================================================================
// OpenAI → Anthropic Conversion
// =============================================================================

/// Convert an OpenAI-format chat request to Anthropic-native format
pub fn convert_openai_to_anthropic(req: &OpenAIChatRequest) -> AnthropicChatRequest {
    let mut system: Option<String> = None;
    let mut messages = Vec::new();

    // Extract system message to top-level system field as required by Anthropic API
    for msg in &req.messages {
        if msg.role == "system" && system.is_none() {
            system = Some(msg.content.clone().unwrap_or_default());
        } else {
            messages.push(AnthropicMessage {
                role: msg.role.clone(),
                content: msg.content.clone().unwrap_or_default(),
            });
        }
    }

    // Anthropic requires max_tokens to be present, so default to 4096 if not specified
    let max_tokens = req.max_tokens.unwrap_or(4096);

    // Convert OpenAI tools to Anthropic tools format
    let tools = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter(|t| t.function.is_some())
            .map(|t| AnthropicTool {
                name: t.function.as_ref().unwrap().name.clone(),
                description: t.function.as_ref().unwrap().description.clone(),
                input_schema: t
                    .function
                    .as_ref()
                    .unwrap()
                    .parameters
                    .clone()
                    .unwrap_or(serde_json::json!({})),
            })
            .collect()
    });

    AnthropicChatRequest {
        model: req.model.clone(),
        messages,
        system,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens,
        stream: req.stream,
        stop_sequences: req.stop.clone(),
        tools,
    }
}

/// Convert an Anthropic-native non-streaming response to OpenAI-format
pub fn convert_anthropic_to_openai(
    resp: &AnthropicChatResponse,
    model: &str,
) -> OpenAIChatResponse {
    // Concatenate all text content blocks
    let content = resp
        .content
        .iter()
        .filter_map(|c| c.text.as_ref())
        .cloned()
        .collect::<Vec<_>>()
        .join("");

    let finish_reason = match resp.stop_reason.as_deref() {
        Some("end_turn") | Some("stop_sequence") => Some("stop".to_string()),
        Some("max_tokens") => Some("length".to_string()),
        Some(other) => Some(other.to_string()),
        None => None,
    };

    OpenAIChatResponse {
        id: resp.id.clone(),
        object: "chat.completion".to_string(),
        created: current_timestamp(),
        model: model.to_string(),
        choices: vec![OpenAIChoice {
            index: 0,
            message: Some(OpenAIMessage {
                role: "assistant".to_string(),
                content: Some(content),
                ..Default::default()
            }),
            delta: None,
            finish_reason,
        }],
        usage: OpenAIUsage {
            prompt_tokens: resp.usage.input_tokens,
            completion_tokens: resp.usage.output_tokens,
            total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
        },
    }
}

/// Convert an Anthropic streaming chunk to OpenAI-format
pub fn convert_anthropic_stream_chunk_to_openai(
    chunk: &AnthropicStreamChunk,
    created: u64,
    model: &str,
) -> OpenAIStreamChunk {
    let text = chunk.delta.as_ref().and_then(|d| d.text.clone());
    let stop_reason = chunk
        .delta
        .as_ref()
        .and_then(|d| d.stop_reason.as_deref())
        .or(chunk.stop_reason.as_deref());

    let finish_reason = match stop_reason {
        Some("end_turn") | Some("stop_sequence") => Some("stop".to_string()),
        Some("max_tokens") => Some("length".to_string()),
        Some(other) => Some(other.to_string()),
        None => None,
    };

    OpenAIStreamChunk {
        id: chunk.id.clone(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![OpenAIStreamChoice {
            index: 0,
            delta: OpenAIDelta {
                role: if text.is_some() {
                    Some("assistant".to_string())
                } else {
                    None
                },
                content: text,
                ..Default::default()
            },
            finish_reason,
        }],
        usage: None,
    }
}

// =============================================================================
// OpenAI → Gemini Conversion
// =============================================================================

/// Convert an OpenAI-format chat request to Gemini-native format
pub fn convert_openai_to_gemini(req: &OpenAIChatRequest) -> GeminiChatRequest {
    let contents = req
        .messages
        .iter()
        .map(|m| {
            // Handle tool role messages - convert to function_response parts
            let parts: Vec<GeminiPart> = if m.role == "tool" {
                vec![GeminiPart {
                    text: None,
                    function_response: Some(GeminiFunctionResponse {
                        name: m.name.clone().unwrap_or_default(),
                        response: m.content.clone().unwrap_or_default(),
                    }),
                }]
            } else {
                vec![GeminiPart {
                    text: m.content.clone(),
                    function_response: None,
                }]
            };

            GeminiContent {
                // Gemini uses "model" instead of "assistant"
                role: if m.role == "assistant" {
                    "model".to_string()
                } else {
                    m.role.clone()
                },
                parts,
            }
        })
        .collect();

    let generation_config = if req.temperature.is_some()
        || req.top_p.is_some()
        || req.max_tokens.is_some()
        || req.stop.is_some()
    {
        Some(GeminiGenerationConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            max_output_tokens: req.max_tokens,
            stop_sequences: req.stop.clone(),
        })
    } else {
        None
    };

    // Convert OpenAI tools to Gemini tools format
    let tools = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter(|t| t.function.is_some())
            .map(|t| GeminiTool {
                function_declarations: vec![GeminiFunctionDeclaration {
                    name: t.function.as_ref().unwrap().name.clone(),
                    description: t.function.as_ref().unwrap().description.clone(),
                    parameters: t.function.as_ref().unwrap().parameters.clone(),
                }],
            })
            .collect()
    });

    GeminiChatRequest {
        contents,
        generation_config,
        tools,
    }
}

/// Convert a Gemini-native non-streaming response to OpenAI-format
pub fn convert_gemini_to_openai(
    resp: &GeminiChatResponse,
    model: &str,
    created: u64,
) -> OpenAIChatResponse {
    let now = if created == 0 {
        current_timestamp()
    } else {
        created
    };
    let id = format!("gemini-{}", now);

    let choices = resp
        .candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            // Concatenate all text parts
            let content = candidate
                .content
                .parts
                .iter()
                .filter_map(|p| p.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join("");

            let finish_reason = match candidate.finish_reason.as_deref() {
                Some("STOP") => Some("stop".to_string()),
                Some("MAX_TOKENS") => Some("length".to_string()),
                Some(other) => Some(other.to_string().to_lowercase()),
                None => None,
            };

            OpenAIChoice {
                index: idx as u32,
                message: Some(OpenAIMessage {
                    role: "assistant".to_string(),
                    content: Some(content),
                    ..Default::default()
                }),
                delta: None,
                finish_reason,
            }
        })
        .collect();

    let usage = match resp.usage_metadata.as_ref() {
        Some(meta) => OpenAIUsage {
            prompt_tokens: meta.prompt_token_count,
            completion_tokens: meta.candidates_token_count,
            total_tokens: meta.total_token_count,
        },
        None => OpenAIUsage::default(),
    };

    OpenAIChatResponse {
        id,
        object: "chat.completion".to_string(),
        created: now,
        model: model.to_string(),
        choices,
        usage,
    }
}

/// Convert a Gemini streaming chunk to OpenAI-format
pub fn convert_gemini_stream_chunk_to_openai(
    chunk: &GeminiStreamChunk,
    id: &str,
    created: u64,
    model: &str,
) -> OpenAIStreamChunk {
    let choices = chunk
        .candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let content = candidate
                .content
                .parts
                .iter()
                .filter_map(|p| p.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join("");

            let finish_reason = match candidate.finish_reason.as_deref() {
                Some("STOP") => Some("stop".to_string()),
                Some("MAX_TOKENS") => Some("length".to_string()),
                Some(other) => Some(other.to_string().to_lowercase()),
                None => None,
            };

            OpenAIStreamChoice {
                index: idx as u32,
                delta: OpenAIDelta {
                    role: if !content.is_empty() {
                        Some("assistant".to_string())
                    } else {
                        None
                    },
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content)
                    },
                    ..Default::default()
                },
                finish_reason,
            }
        })
        .collect();

    OpenAIStreamChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices,
        usage: None,
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Get current timestamp in seconds since UNIX epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// =============================================================================
// OpenAI Responses API ↔ Chat Completions Conversion
// =============================================================================

/// Convert a Responses API request to Chat Completions API request
pub fn convert_responses_to_chat(req: &ResponsesRequest) -> OpenAIChatRequest {
    let mut messages = Vec::new();

    // Add instructions as system message first
    if let Some(instructions) = &req.instructions {
        messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: Some(instructions.clone()),
            ..Default::default()
        });
    }

    // Convert input to messages
    match &req.input {
        Some(ResponseInput::String(s)) => messages.push(OpenAIMessage {
            role: "user".to_string(),
            content: Some(s.clone()),
            ..Default::default()
        }),
        Some(ResponseInput::Messages(msgs)) => messages.extend(msgs.clone()),
        Some(ResponseInput::Raw(value)) => {
            // Try to parse raw JSON as an array of messages
            if let Some(arr) = value.as_array() {
                let parsed_messages: Vec<OpenAIMessage> = arr
                    .iter()
                    .map(|v| {
                        // Try to deserialize each item as OpenAIMessage
                        serde_json::from_value(v.clone()).unwrap_or_else(|_| {
                            // Fallback: create a simple user message
                            OpenAIMessage {
                                role: "user".to_string(),
                                content: Some(v.to_string()),
                                ..Default::default()
                            }
                        })
                    })
                    .collect();
                messages.extend(parsed_messages);
            } else {
                // Not an array - create a single message with string representation
                messages.push(OpenAIMessage {
                    role: "user".to_string(),
                    content: Some(value.to_string()),
                    ..Default::default()
                });
            }
        }
        None => {}
    };

    // Filter out tools without function field and ensure parameters has type: "object"
    let tools = req.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter(|t| t.function.is_some())
            .map(|t| {
                let mut tool = t.clone();
                if let Some(ref mut function) = tool.function
                    && let Some(ref mut params) = function.parameters
                    && params.is_object()
                    && !params.as_object().unwrap().contains_key("type")
                    && let Some(obj) = params.as_object_mut()
                {
                    obj.insert("type".to_string(), serde_json::Value::String("object".to_string()));
                }
                tool
            })
            .collect()
    });

    // Convert text.format to response_format
    let response_format = req.text.as_ref().and_then(|t| t.format.clone());

    // Extract reasoning_effort from reasoning
    let reasoning_effort = req.reasoning.as_ref().and_then(|r| r.effort.clone());

    OpenAIChatRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_output_tokens,
        stream: req.stream,
        tools,
        tool_choice: req.tool_choice.clone(),
        response_format,
        reasoning_effort,
        ..Default::default()
    }
}

/// Convert a Chat Completions response to Responses API format
pub fn convert_chat_to_responses(resp: &OpenAIChatResponse, created_at: i64) -> ResponsesResponse {
    // Extract message content from first choice
    let (content_text, finish_reason) = resp
        .choices
        .first()
        .map(|choice| {
            let text = choice
                .message
                .as_ref()
                .and_then(|m| m.content.clone())
                .unwrap_or_default();
            (text, choice.finish_reason.clone())
        })
        .unwrap_or_default();

    // Determine status based on finish_reason
    let status = match finish_reason.as_deref() {
        Some("stop") | Some("length") | Some("tool_calls") => "completed",
        _ => "completed",
    };

    // Convert usage
    let usage = ResponseUsage {
        input_tokens: resp.usage.prompt_tokens,
        input_tokens_details: None,
        output_tokens: resp.usage.completion_tokens,
        output_tokens_details: None,
        total_tokens: resp.usage.total_tokens,
    };

    ResponsesResponse {
        id: resp.id.clone(),
        object: "response".to_string(),
        created_at,
        model: resp.model.clone(),
        error: None,
        incomplete_details: None,
        instructions: None,
        metadata: None,
        output: vec![ResponseOutputItem {
            output_type: "message".to_string(),
            id: Some("msg_0".to_string()),
            status: Some("completed".to_string()),
            role: Some("assistant".to_string()),
            content: Some(vec![ResponseContentPart {
                content_type: "output_text".to_string(),
                text: Some(content_text),
                annotations: None,
            }]),
        }],
        parallel_tool_calls: None,
        temperature: None,
        tool_choice: None,
        tools: None,
        top_p: None,
        max_output_tokens: None,
        previous_response_id: None,
        reasoning: None,
        status: status.to_string(),
        text: None,
        truncation: None,
        usage: Some(usage),
        user: None,
        store: None,
    }
}

/// Convert streaming Chat Completions chunk to Responses API format
/// Returns the event type and JSON data
pub fn convert_chat_stream_chunk_to_responses(
    chunk: &OpenAIStreamChunk,
    response_id: &str,
    created_at: i64,
) -> Vec<(String, serde_json::Value)> {
    let mut events = Vec::new();
    let item_id_prefix = &response_id[0..std::cmp::min(8, response_id.len())];

    for choice in &chunk.choices {
        // Handle text content delta
        if let Some(ref content) = choice.delta.content
            && !content.is_empty()
        {
            let delta_event = serde_json::json!({
                "type": "response.output_text.delta",
                "item_id": format!("item_{}", item_id_prefix),
                "output_index": 0,
                "content_index": 0,
                "delta": content,
                "created_at": created_at
            });
            events.push(("response.output_text.delta".to_string(), delta_event));
        }

        // Handle tool calls delta
        if let Some(ref tool_calls) = choice.delta.tool_calls {
            for tool_call in tool_calls {
                let call_index = tool_call.index.unwrap_or(0);
                let output_index = call_index + 1;  // +1 because text is index 0
                if let Some(ref function) = tool_call.function {
                    // Send function call arguments delta
                    if let Some(ref args) = function.arguments
                        && !args.is_empty()
                    {
                        let delta_event = serde_json::json!({
                            "type": "response.function_call_arguments.delta",
                            "item_id": format!("func_{}_{}", item_id_prefix, output_index),
                            "output_index": output_index,
                            "delta": args,
                            "created_at": created_at
                        });
                        events.push(("response.function_call_arguments.delta".to_string(), delta_event));
                    }
                }
            }
        }
    }

    events
}

/// Create initial response.created event for streaming
pub fn create_response_created_event(
    id: &str,
    model: &str,
    created_at: i64,
) -> (String, serde_json::Value) {
    let event = serde_json::json!({
        "type": "response.created",
        "response": {
            "id": id,
            "object": "response",
            "created_at": created_at,
            "model": model,
            "status": "in_progress",
            "output": [],
            "usage": {}
        }
    });
    ("response.created".to_string(), event)
}

/// Create response.output_item.done event
pub fn create_response_output_item_done_event(
    id: &str,
    created_at: i64,
) -> (String, serde_json::Value) {
    let event = serde_json::json!({
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {
            "id": id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": []
        },
        "created_at": created_at
    });
    ("response.output_item.done".to_string(), event)
}

/// Format a Responses API event as SSE line
pub fn format_responses_sse_event(event_type: &str, data: &serde_json::Value) -> String {
    format!("event: {}\ndata: {}\n\n", event_type, serde_json::to_string(data).unwrap())
}

/// Format the final [DONE] SSE line
pub fn format_responses_sse_done() -> String {
    "data: [DONE]\n\n".to_string()
}
