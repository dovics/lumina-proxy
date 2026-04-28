//! Tool call parsing and aggregation utilities

use crate::types::{OpenAIToolCall, OpenAIToolCallFunction};

// =============================================================================
// Tool Call Aggregation Helpers
// =============================================================================

/// Aggregates streaming tool_call chunks into a complete tool_call
/// Tool calls come in pieces: id, function.name, function.arguments across multiple chunks
pub fn aggregate_tool_calls(tool_calls: &[OpenAIToolCall]) -> Vec<OpenAIToolCall> {
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
///
/// Note: This function handles multiple format variations:
/// 1. Full format with section and call wrappers
/// 2. Only <|tool_call_argument_begin|> and <|tool_calls_section_end|> markers
pub fn parse_moonlight_tool_calls(content: &str) -> Vec<OpenAIToolCall> {
    let mut tool_calls = Vec::new();

    tracing::debug!(
        content = %content,
        has_tool_call_begin = %content.contains("<|tool_call_begin|>"),
        has_tool_call_argument_begin = %content.contains("<|tool_call_argument_begin|>"),
        has_tool_calls_section_end = %content.contains("<|tool_calls_section_end|>"),
        "parse_moonlight_tool_calls: analyzing content"
    );

    if content.contains("<|tool_call_begin|>") {
        tracing::debug!("parse_moonlight_tool_calls: using full format (with wrappers)");
        let (section_start, section_end) =
            if let Some(start) = content.find("<|tool_calls_section_begin|>") {
                let end = content
                    .find("<|tool_calls_section_end|>")
                    .unwrap_or(content.len());
                (start + "<|tool_calls_section_begin|>".len(), end)
            } else {
                (0, content.len())
            };

        let section_content = &content[section_start..section_end];
        let mut index: u32 = 0;

        let mut remaining = section_content;
        while let Some(call_start) = remaining.find("<|tool_call_begin|>") {
            let after_call_start = &remaining[call_start + "<|tool_call_begin|>".len()..];
            if let Some(call_end) = after_call_start.find("<|tool_call_end|>") {
                let call_content = &after_call_start[..call_end];
                if let Some(tool_call) = extract_moonlight_tool_call(call_content, index) {
                    tracing::debug!(
                        "parse_moonlight_tool_calls: extracted tool_call name={:?} args={:?}",
                        tool_call.function.as_ref().and_then(|f| f.name.clone()),
                        tool_call
                            .function
                            .as_ref()
                            .and_then(|f| f.arguments.clone())
                    );
                    tool_calls.push(tool_call);
                    index += 1;
                }
                remaining = &after_call_start[call_end + "<|tool_call_end|>".len()..];
            } else {
                break;
            }
        }
    } else if content.contains("<|tool_call_argument_begin|>") {
        tracing::debug!("parse_moonlight_tool_calls: using simplified format");
        let section_end = content
            .find("<|tool_calls_section_end|>")
            .unwrap_or(content.len());

        if let Some(arg_pos) = content.find("<|tool_call_argument_begin|>") {
            let before_arg = &content[0..arg_pos];
            let after_arg = &content[arg_pos + "<|tool_call_argument_begin|>".len()..section_end];

            let tool_id = before_arg.trim();
            let arguments = after_arg.trim();

            tracing::debug!(
                "parse_moonlight_tool_calls: simplified format - tool_id={} arguments={}",
                tool_id,
                arguments
            );

            if let Some(tool_call) =
                extract_moonlight_tool_call_from_id_and_args(tool_id, arguments, 0)
            {
                tracing::debug!(
                    "parse_moonlight_tool_calls: simplified format extracted name={:?} args={:?}",
                    tool_call.function.as_ref().and_then(|f| f.name.clone()),
                    tool_call
                        .function
                        .as_ref()
                        .and_then(|f| f.arguments.clone())
                );
                tool_calls.push(tool_call);
            }
        }
    } else {
        tracing::debug!("parse_moonlight_tool_calls: no recognized markers found");
    }

    tracing::debug!(
        "parse_moonlight_tool_calls: returning {} tool_calls",
        tool_calls.len()
    );
    tool_calls
}

/// Extract a single tool call from a tool ID and arguments string
fn extract_moonlight_tool_call_from_id_and_args(
    tool_id: &str,
    arguments: &str,
    index: u32,
) -> Option<OpenAIToolCall> {
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

    let id = format!("call_{}", generate_random_id());

    Some(OpenAIToolCall {
        index: Some(index),
        id: Some(id),
        r#type: Some("function".to_string()),
        function: Some(OpenAIToolCallFunction {
            name: Some(func_name.to_string()),
            arguments: Some(arguments.to_string()),
        }),
    })
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

    let id = format!("call_{}", generate_random_id());

    Some(OpenAIToolCall {
        index: Some(index),
        id: Some(id),
        r#type: Some("function".to_string()),
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
pub fn strip_moonlight_tool_markers(content: &str) -> String {
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
