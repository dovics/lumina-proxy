# Moonlight Tool Call Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Moonlight provider support with tool call parsing from content markers

**Architecture:** Add Moonlight as a new ProviderType that handles streaming responses by parsing special markers (`<|tool_calls_section_begin|>`, etc.) from the content field and converting them to standard OpenAI tool_calls format.

**Tech Stack:** Rust (Axum, Serde), lumina-proxy codebase

---

## File Structure

| File | Changes |
|------|---------|
| `src/config.rs` | Add `Moonlight` variant to `ProviderType` enum |
| `src/convert.rs` | Add `convert_openai_to_moonlight()` passthrough function |
| `src/proxy.rs` | Add Moonlight branch in `handle_streaming()`, add `parse_moonlight_tool_calls()` function |
| `config.yaml` | Add Moonlight provider example |
| `tests/conversion_tests.rs` | Add Moonlight tool call parsing tests |

---

## Task 1: Add Moonlight to ProviderType Enum

**Files:**
- Modify: `src/config.rs:21-37`

- [ ] **Step 1: Add Moonlight variant to ProviderType enum**

Read line 21-37 of `src/config.rs`, then add:

```rust
    /// Google Gemini API
    #[serde(rename = "gemini")]
    Gemini,
    /// Moonlight (月光) API
    #[serde(rename = "moonlight")]
    Moonlight,
    /// OpenAI-compatible API (e.g., vLLM, llama-cpp-python, etc.)
    #[serde(rename = "openai-compatible")]
    OpenAiCompatible,
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build 2>&1 | head -50`
Expected: Should compile without errors related to ProviderType

- [ ] **Step 3: Commit**

```bash
git add src/config.rs
git commit -m "feat: Add Moonlight provider type

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Add Moonlight Passthrough Conversion Function

**Files:**
- Modify: `src/convert.rs`

- [ ] **Step 1: Add convert_openai_to_moonlight function**

Read the end of `src/convert.rs` (around line 739) to find where to add the new function. Add after the last function:

```rust
/// Convert an OpenAI-format chat request to Moonlight (passthrough - Moonlight accepts OpenAI format)
pub fn convert_openai_to_moonlight(req: &OpenAIChatRequest) -> OpenAIChatRequest {
    req.clone()
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build 2>&1 | head -50`
Expected: Should compile without errors

- [ ] **Step 3: Commit**

```bash
git add src/convert.rs
git commit -m "feat: Add Moonlight passthrough conversion

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Add rand Dependency

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Check if rand is already a dependency**

Run: `grep -n "rand" Cargo.toml`
Expected: If found, skip to Task 4. If not found, continue.

- [ ] **Step 2: Add rand dependency**

Read `Cargo.toml` to find the `[dependencies]` section. Add:
```toml
rand = "0.8"
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build 2>&1 | head -30`
Expected: Dependencies resolve without errors

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml
git commit -m "chore: Add rand dependency for Moonlight tool call ID generation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Add Moonlight Tool Call Parsing Functions

**Files:**
- Modify: `src/proxy.rs`

- [ ] **Step 1: Add parse_moonlight_tool_calls function**

Read `src/proxy.rs` around line 145 (where `aggregate_tool_calls` is defined) to understand the context. Add the following new function after `aggregate_tool_calls`:

```rust
/// Parse tool calls from Moonlight's special marker format embedded in content field.
///
/// Marker format:
/// - `<|tool_calls_section_begin|>` / `<|tool_calls_section_end|>` - wraps all tool calls
/// - `<|tool_call_begin|>` / `<|tool_call_end|>` - wraps each tool call
/// - `<|tool_call_argument_begin|>` - separates tool ID from arguments
///
/// Tool ID format: `functions.{func_name}:{idx}`
fn parse_moonlight_tool_calls(content: &str) -> Vec<OpenAIToolCall> {
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
        tracing::warn!("Malformed tool call segment: expected 2 parts separated by <|tool_call_argument_begin|>");
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
        tracing::warn!("Malformed tool ID '{}': missing 'functions.' prefix", tool_id);
        return None;
    };

    // Generate a random ID
    let id = format!("call_{}", generate_random_id());

    Some(OpenAIToolCall {
        index: Some(index),
        id,
        type_: Some("function".to_string()),
        function: OpenAIToolCallFunction {
            name: Some(func_name.to_string()),
            arguments: Some(arguments.to_string()),
        },
    })
}

/// Generate a random 12-character alphanumeric ID
fn generate_random_id() -> String {
    use std::iter;
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::Rng::random_u64(); // Simple random, good enough for IDs
    iter::repeat_with(|| {
        let idx = (rng % CHARSET.len() as u64) as usize;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build 2>&1 | head -100`
Expected: Should compile without errors

- [ ] **Step 3: Commit**

```bash
git add src/proxy.rs
git commit -m "feat: Add Moonlight tool call parsing functions

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Add Moonlight Branch in handle_streaming

**Files:**
- Modify: `src/proxy.rs`

There are **TWO** `match provider_type` blocks in `handle_streaming` that both need Moonlight added:

1. **Request building block** (around lines 430-468): Builds the request to send to upstream
2. **Response chunk processing block** (around lines 590-660): Processes incoming streaming chunks

- [ ] **Step 1: Read handle_streaming function to find both match blocks**

Read `src/proxy.rs` lines 410-660 to understand both match blocks.

- [ ] **Step 2: Add Moonlight branch in the REQUEST building match block (first match)**

In the first `match provider_type` block (around line 430), add:

```rust
ProviderType::Moonlight => serde_json::to_vec(&convert_openai_to_moonlight(&streaming_req))
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Failed to serialize Moonlight request: {}", e) })),
        )
    }),
```

**Important:** Add `Moonlight` to the response_id format match block (around line 500-510). Find the match that creates the `response_id` string and add:

```rust
match route.provider_type {
    ProviderType::OpenAi => "chatcmpl",
    ProviderType::Ollama => "ollama",
    ProviderType::Anthropic => "anthropic",
    ProviderType::Gemini => "gemini",
    ProviderType::Moonlight => "moonlight",  // Add this line
    ProviderType::OpenAiCompatible => "chatcmpl",
},
```

- [ ] **Step 3: Add Moonlight branch in the RESPONSE chunk processing match block (second match)**

In the second `match provider_type` block for processing response chunks (around line 590), add a new branch using this code:

```rust
ProviderType::Moonlight => {
    // Moonlight accepts OpenAI format, but returns tool calls embedded in content
    // as special markers that need parsing
    match serde_json::from_str::<OpenAIStreamChunk>(data) {
        Ok(mut openai_chunk) => {
            let mut has_content = false;

            // Process each choice
            for choice in &openai_chunk.choices {
                // Count content tokens
                if let Some(content) = &choice.delta.content
                    && !content.is_empty()
                {
                    counter.add_delta(content);
                    has_content = true;
                }

                // Parse tool calls from content markers
                if let Some(content) = &choice.delta.content {
                    let parsed_tool_calls = parse_moonlight_tool_calls(content);
                    if !parsed_tool_calls.is_empty() {
                        // We found tool calls in the content
                        // Add them to the token counter
                        for tc in &parsed_tool_calls {
                            if let Some(ref func) = tc.function {
                                if let Some(ref name) = func.name {
                                    counter.add_delta(name);
                                }
                                if let Some(ref args) = func.arguments {
                                    counter.add_delta(args);
                                }
                            }
                        }
                        has_content = true;
                    }
                }
            }

            // If we found tool calls in content, extract them and keep any text before/after markers
            for choice in &mut openai_chunk.choices {
                if let Some(content) = &choice.delta.content.clone() {
                    let parsed_tool_calls = parse_moonlight_tool_calls(&content);
                    if !parsed_tool_calls.is_empty() {
                        // Strip tool call markers from content but keep surrounding text
                        let text_content = strip_moonlight_tool_markers(&content);
                        if !text_content.trim().is_empty() {
                            choice.delta.content = Some(text_content);
                        } else {
                            choice.delta.content = None;
                        }
                        choice.delta.tool_calls = Some(parsed_tool_calls);
                    }
                }
            }

            if !has_content {
                tracing::trace!(
                    "Moonlight chunk with no countable content: {}",
                    data.len().min(500)
                );
            }

            openai_chunk.id = id.clone();
            openai_chunk.model = model.clone();
            Ok(openai_chunk)
        }
        Err(e) => {
            tracing::warn!("Failed to parse Moonlight stream chunk: {}", e);
            Err(format!("Failed to parse Moonlight stream chunk: {}", e))
        }
    }
}
```

- [ ] **Step 4: Verify compilation**

Run: `cargo build 2>&1 | head -100`
Expected: Should compile. Fix any missing matches.

- [ ] **Step 5: Commit**

```bash
git add src/proxy.rs
git commit -m "feat: Add Moonlight branch in handle_streaming

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Add Moonlight Configuration Example

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add Moonlight example to config.yaml**

Read `config.yaml` around line 60 (after the ollama example). Add:

```yaml
  # Moonlight (月光) API example
  - model_name: "your-moonlight-model"
    provider_type: "moonlight"
    url: "https://api.moonlight.example.com/v1/chat/completions"
    api_key: "your-moonlight-api-key"
    enabled: true
```

- [ ] **Step 2: Also update the comment listing provider types**

Read line 34 of `config.yaml`. Update the comment to include `moonlight`:
```yaml
    provider_type: "openai"          # openai | ollama | anthropic | gemini | moonlight | openai-compatible
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "docs: Add Moonlight provider example to config

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Add Moonlight Tool Call Parsing Tests

**Files:**
- Modify: `tests/conversion_tests.rs`

- [ ] **Step 1: Write failing tests for Moonlight tool call parsing**

Read `tests/conversion_tests.rs` to understand the test structure and naming conventions. Add these tests:

```rust
#[test]
fn test_parse_moonlight_tool_calls_basic() {
    use crate::proxy::parse_moonlight_tool_calls;

    let content = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"北京\"}<|tool_call_end|><|tool_calls_section_end|>";
    let tool_calls = parse_moonlight_tool_calls(content);

    assert_eq!(tool_calls.len(), 1);
    let tc = &tool_calls[0];
    assert_eq!(tc.index, Some(0));
    assert!(tc.id.starts_with("call_"));
    assert_eq!(tc.function.name, Some("get_weather".to_string()));
    assert_eq!(tc.function.arguments, Some("{\"location\": \"北京\"}".to_string()));
}

#[test]
fn test_parse_moonlight_tool_calls_multiple() {
    use crate::proxy::parse_moonlight_tool_calls;

    let content = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"北京\"}<|tool_call_end|><|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>{\"timezone\": \"Asia/Shanghai\"}<|tool_call_end|><|tool_calls_section_end|>";
    let tool_calls = parse_moonlight_tool_calls(content);

    assert_eq!(tool_calls.len(), 2);

    assert_eq!(tool_calls[0].function.name, Some("get_weather".to_string()));
    assert_eq!(tool_calls[0].index, Some(0));

    assert_eq!(tool_calls[1].function.name, Some("get_time".to_string()));
    assert_eq!(tool_calls[1].index, Some(1));
}

#[test]
fn test_parse_moonlight_tool_calls_with_text() {
    use crate::proxy::parse_moonlight_tool_calls;

    // Text before and after markers should be ignored
    let content = "Some text before<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"北京\"}<|tool_call_end|><|tool_calls_section_end|>Some text after";
    let tool_calls = parse_moonlight_tool_calls(content);

    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, Some("get_weather".to_string()));
}

#[test]
fn test_parse_moonlight_tool_calls_empty() {
    use crate::proxy::parse_moonlight_tool_calls;

    // No tool calls section
    let content = "Just regular text content";
    let tool_calls = parse_moonlight_tool_calls(content);
    assert!(tool_calls.is_empty());

    // Empty section
    let content = "<|tool_calls_section_begin|><|tool_calls_section_end|>";
    let tool_calls = parse_moonlight_tool_calls(content);
    assert!(tool_calls.is_empty());
}

#[test]
fn test_parse_moonlight_tool_calls_malformed() {
    use crate::proxy::parse_moonlight_tool_calls;

    // Missing end marker - should gracefully handle
    let content = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"location\": \"北京\"}";
    let tool_calls = parse_moonlight_tool_calls(content);
    assert!(tool_calls.is_empty());

    // Missing argument separator
    let content = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0{\"location\": \"北京\"}<|tool_call_end|><|tool_calls_section_end|>";
    let tool_calls = parse_moonlight_tool_calls(content);
    assert!(tool_calls.is_empty());
}
```

**Note:** The tests reference `crate::proxy::parse_moonlight_tool_calls`. Since the function is `pub(crate)`, it will be accessible from tests within the crate.

- [ ] **Step 2: Run tests to verify they fail (function not visible or doesn't exist yet)**

Run: `cargo test test_parse_moonlight 2>&1`
Expected: Test should compile and fail with assertion errors once function is public

- [ ] **Step 3: Make parse_moonlight_tool_calls pub(crate) if needed**

If tests can't access the function, change the function signature from `fn` to `pub(crate) fn`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test test_parse_moonlight 2>&1`
Expected: All Moonlight tests pass

- [ ] **Step 5: Run all tests to ensure no regressions**

Run: `cargo test 2>&1`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add tests/conversion_tests.rs src/proxy.rs
git commit -m "test: Add Moonlight tool call parsing tests

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Final Verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -- -D warnings 2>&1 | head -100`
Expected: No warnings or errors

- [ ] **Step 2: Run format check**

Run: `cargo fmt --check 2>&1`
Expected: No formatting issues (run `cargo fmt` if needed)

- [ ] **Step 3: Run all tests**

Run: `cargo test 2>&1`
Expected: All tests pass

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: Address clippy/format/test issues

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Success Criteria

1. ✅ Moonlight provider type is recognized in configuration
2. ✅ Streaming responses with tool call markers are correctly parsed
3. ✅ Tool calls are converted to standard OpenAI format with correct function names and arguments
4. ✅ Text content surrounding tool call markers is handled (tool calls extracted, text preserved)
5. ✅ All existing tests continue to pass
6. ✅ New tests for Moonlight parsing pass
7. ✅ Clippy passes with `-D warnings`
8. ✅ Code is properly formatted