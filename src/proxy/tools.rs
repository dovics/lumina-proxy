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
