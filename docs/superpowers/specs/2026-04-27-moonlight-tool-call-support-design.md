# Moonlight Tool Call Support - Design Specification

## Overview

Add support for Moonlight (月光) provider's tool call format in lumina-proxy. Moonlight's streaming responses embed tool call information as special markers within the `content` field, requiring parsing and conversion to standard OpenAI `tool_calls` format.

## Problem Statement

Moonlight-K2 generates tool call requests wrapped with markers:
- `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>` - wraps all tool calls
- `<|tool_call_begin|>` / `<|tool_call_end|>` - wraps each individual tool call
- `<|tool_call_argument_begin|>` - separates tool ID from arguments

The tool ID format is `functions.{func_name}:{idx}`, from which the function name is parsed.

## Architecture

```
Client Request (OpenAI tool format)
        │
        ▼
Moonlight Backend Streaming Response
    content contains: "...text<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"北京"}<|tool_call_end|><|tool_calls_section_end|>..."
        │
        ▼
Parse content, extract tool call markers
        │
        ▼
Convert to standard OpenAI tool_calls format
        │
        ▼
Return to client
```

## Implementation Plan

### 1. Add Moonlight ProviderType

**File: `src/types.rs`**

Add `Moonlight` to `ProviderType` enum:
```rust
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProviderType {
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "ollama")]
    Ollama,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "gemini")]
    Gemini,
    #[serde(rename = "moonlight")]
    Moonlight,
    #[serde(rename = "openai-compatible")]
    OpenAiCompatible,
}
```

### 2. Add Moonlight Request Conversion (Passthrough)

**File: `src/convert.rs`**

Request conversion is identity (passthrough) since Moonlight accepts OpenAI format:
```rust
pub fn convert_openai_to_moonlight(req: &OpenAIChatRequest) -> OpenAIChatRequest {
    req.clone()
}
```

### 3. Parse Tool Calls from Content in Streaming Response

**File: `src/proxy.rs`**

Add `Moonlight` branch in `handle_streaming()` function's match statement for `provider_type`.

**Parsing Logic:**
1. Detect `<|tool_calls_section_begin|>` in content
2. Extract content between `<|tool_call_begin|>` and `<|tool_call_end|>`
3. Split by `<|tool_call_argument_begin|>` to get:
   - Tool ID (format: `functions.{func_name}:{idx}`)
   - Arguments JSON string
4. Parse function name from tool ID
5. Create `OpenAIToolCall` structure

**Key Functions to Add:**

```rust
/// Parse tool calls from Moonlight's special marker format in content
fn parse_moonlight_tool_calls(content: &str) -> Vec<OpenAIToolCall>

/// Extract a single tool call from content segment
fn extract_moonlight_tool_call(segment: &str, index: u32) -> Option<OpenAIToolCall>
```

### 4. Update Configuration

**File: `config.yaml`**

Add Moonlight provider example:
```yaml
routes:
  - model_name: "your-model"
    provider_type: "moonlight"
    url: "https://api.moonlight.example.com/v1/chat/completions"
    api_key: "your-api-key"
    enabled: true
```

## Tool Call Marker Format

### Input (Moonlight content field):
```
Some text before<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "北京", "unit": "celsius"}<|tool_call_end|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{"timezone": "Asia/Shanghai"}<|tool_call_end|><|tool_calls_section_end|>Some text after
```

### Output (Standard OpenAI tool_calls):
```json
{
  "tool_calls": [
    {
      "index": 0,
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"北京\", \"unit\": \"celsius\"}"
      }
    },
    {
      "index": 1,
      "id": "call_def456",
      "type": "function",
      "function": {
        "name": "get_time",
        "arguments": "{\"timezone\": \"Asia/Shanghai\"}"
      }
    }
  ]
}
```

**Note:** The `id` field is generated using format `call_{random_12_alphanumeric_chars}` since Moonlight doesn't provide one in this format. Use `format!("call_{}", generate_random_id())` where `generate_random_id()` produces 12 random alphanumeric characters.

## Edge Cases

1. **Nested markers in arguments**: Arguments are JSON, may contain similar-looking strings. Parse strictly by marker positions.
2. **Multiple tool calls in one content chunk**: Handle all occurrences between section markers.
3. **Tool calls split across chunks**: Buffer content until `tool_calls_section_end` is seen.
4. **Text before/after markers**: Preserve text content separately from tool calls.
5. **Empty tool calls section**: Return empty vector.
6. **Malformed markers**: Skip malformed segments, log warning.

## Testing

Add tests in `tests/conversion_tests.rs`:
- `test_parse_moonlight_tool_calls_basic` - single tool call
- `test_parse_moonlight_tool_calls_multiple` - multiple tool calls
- `test_parse_moonlight_tool_calls_with_text` - text before/after markers
- `test_parse_moonlight_tool_calls_malformed` - handle gracefully

## Files to Modify

| File | Changes |
|------|---------|
| `src/types.rs` | Add `Moonlight` to `ProviderType` enum |
| `src/convert.rs` | Add `convert_openai_to_moonlight()` function |
| `src/proxy.rs` | Add Moonlight branch in streaming handler, add parsing functions |
| `config.yaml` | Add Moonlight provider example |
| `tests/conversion_tests.rs` | Add Moonlight tool call parsing tests |

**Note:** `config.rs` does not require modification - provider type validation is enum-based and automatically handles new variants.

## Success Criteria

1. Moonlight provider type is recognized in configuration
2. Streaming responses with tool call markers are correctly parsed
3. Tool calls are converted to standard OpenAI format with correct function names and arguments
4. Text content surrounding tool call markers is preserved
5. All existing tests continue to pass
6. New tests for Moonlight parsing pass