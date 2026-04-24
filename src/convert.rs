//! Format conversion functions between OpenAI standard and provider-native formats

use crate::types::*;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// OpenAI → Ollama Conversion
// =============================================================================

/// Convert an OpenAI-format chat request to Ollama-native format
pub fn convert_openai_to_ollama(req: &OpenAIChatRequest) -> OllamaChatRequest {
    let messages = req.messages.iter()
        .map(|m| OllamaMessage {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();

    OllamaChatRequest {
        model: req.model.clone(),
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        num_predict: req.max_tokens,
        stream: req.stream,
        stop: req.stop.clone(),
    }
}

/// Convert an Ollama-native non-streaming response to OpenAI-format
pub fn convert_ollama_to_openai(
    resp: &OllamaChatResponse,
    model: &str,
) -> OpenAIChatResponse {
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

    AnthropicChatRequest {
        model: req.model.clone(),
        messages,
        system,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens,
        stream: req.stream,
        stop_sequences: req.stop.clone(),
    }
}

/// Convert an Anthropic-native non-streaming response to OpenAI-format
pub fn convert_anthropic_to_openai(
    resp: &AnthropicChatResponse,
    model: &str,
) -> OpenAIChatResponse {
    // Concatenate all text content blocks
    let content = resp.content.iter()
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
    let stop_reason = chunk.delta.as_ref()
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
    }
}

// =============================================================================
// OpenAI → Gemini Conversion
// =============================================================================

/// Convert an OpenAI-format chat request to Gemini-native format
pub fn convert_openai_to_gemini(req: &OpenAIChatRequest) -> GeminiChatRequest {
    let contents = req.messages.iter()
        .map(|m| GeminiContent {
            // Gemini uses "model" instead of "assistant"
            role: if m.role == "assistant" { "model".to_string() } else { m.role.clone() },
            parts: vec![GeminiPart {
                text: m.content.clone(),
            }],
        })
        .collect();

    let generation_config = if req.temperature.is_some() || req.top_p.is_some() || req.max_tokens.is_some() || req.stop.is_some() {
        Some(GeminiGenerationConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            max_output_tokens: req.max_tokens,
            stop_sequences: req.stop.clone(),
        })
    } else {
        None
    };

    GeminiChatRequest {
        contents,
        generation_config,
    }
}

/// Convert a Gemini-native non-streaming response to OpenAI-format
pub fn convert_gemini_to_openai(
    resp: &GeminiChatResponse,
    model: &str,
    created: u64,
) -> OpenAIChatResponse {
    let now = if created == 0 { current_timestamp() } else { created };
    let id = format!("gemini-{}", now);

    let choices = resp.candidates.iter()
        .enumerate()
        .map(|(idx, candidate)| {
            // Concatenate all text parts
            let content = candidate.content.parts.iter()
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
    let choices = chunk.candidates.iter()
        .enumerate()
        .map(|(idx, candidate)| {
            let content = candidate.content.parts.iter()
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
                    content: if content.is_empty() { None } else { Some(content) },
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
