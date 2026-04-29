use lumina::convert::{
    convert_anthropic_stream_chunk_to_openai, convert_anthropic_to_openai,
    convert_gemini_stream_chunk_to_openai, convert_gemini_to_openai,
    convert_ollama_stream_chunk_to_openai, convert_ollama_to_openai, convert_openai_to_anthropic,
    convert_openai_to_gemini, convert_openai_to_ollama,
};
use lumina::types::{
    AnthropicChatResponse, AnthropicContent, AnthropicDelta, AnthropicStreamChunk, AnthropicUsage,
    GeminiCandidate, GeminiChatResponse, GeminiContent, GeminiPart, GeminiStreamChunk,
    GeminiUsageMetadata, MessageContent, OllamaChatResponse, OllamaDelta, OllamaMessage,
    OllamaStreamChunk, OpenAIChatRequest, OpenAIMessage, OpenAITool, OpenAIToolCall,
    OpenAIToolCallFunction, OpenAIToolFunction,
};

fn openai_message(role: &str, content: &str) -> OpenAIMessage {
    OpenAIMessage {
        role: role.to_string(),
        content: Some(MessageContent::String(content.to_string())),
        ..Default::default()
    }
}

#[test]
fn test_openai_to_ollama_conversion() {
    let openai_req = OpenAIChatRequest {
        model: "llama3:8b".to_string(),
        messages: vec![openai_message("user", "Hello!")],
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(100),
        stream: Some(true),
        stop: Some(vec!["<stop>".to_string()]),
        presence_penalty: Some(0.1),
        frequency_penalty: Some(0.2),
        ..Default::default()
    };

    let ollama_req = convert_openai_to_ollama(&openai_req);
    assert_eq!(ollama_req.model, "llama3:8b");
    assert_eq!(ollama_req.messages[0].role, "user");
    assert_eq!(ollama_req.messages[0].content, "Hello!");
    assert_eq!(ollama_req.temperature, Some(0.7));
    assert_eq!(ollama_req.top_p, Some(0.9));
    assert_eq!(ollama_req.num_predict, Some(100));
    assert_eq!(ollama_req.stream, Some(true));
    assert_eq!(ollama_req.stop, Some(vec!["<stop>".to_string()]));
}

#[test]
fn test_ollama_to_openai_response_conversion() {
    let ollama_resp = OllamaChatResponse {
        model: "llama3:8b".to_string(),
        created_at: "2024-01-01T00:00:00Z".to_string(),
        message: OllamaMessage {
            role: "assistant".to_string(),
            content: "Hello! How can I help you today?".to_string(),
        },
        done: true,
        eval_count: Some(20),
        prompt_eval_count: Some(5),
    };

    let openai_resp = convert_ollama_to_openai(&ollama_resp, "test-model");
    assert_eq!(openai_resp.model, "test-model");
    assert_eq!(openai_resp.choices.len(), 1);
    assert_eq!(openai_resp.choices[0].index, 0);
    assert!(openai_resp.choices[0].message.is_some());
    let message = openai_resp.choices[0].message.as_ref().unwrap();
    assert_eq!(message.role, "assistant");
    assert_eq!(
        message.content.as_ref().map(|c| c.as_string()).as_deref(),
        Some("Hello! How can I help you today?")
    );
    assert_eq!(openai_resp.usage.prompt_tokens, 5);
    assert_eq!(openai_resp.usage.completion_tokens, 20);
    assert_eq!(openai_resp.usage.total_tokens, 25);
    assert_eq!(
        openai_resp.choices[0].finish_reason,
        Some("stop".to_string())
    );
}

#[test]
fn test_ollama_stream_chunk_to_openai() {
    let ollama_chunk = OllamaStreamChunk {
        model: "llama3:8b".to_string(),
        created_at: "2024-01-01T00:00:00Z".to_string(),
        delta: Some(OllamaDelta {
            content: "Hello".to_string(),
        }),
        message: None,
        done: false,
        eval_count: None,
        prompt_eval_count: None,
    };

    let openai_chunk =
        convert_ollama_stream_chunk_to_openai(&ollama_chunk, "test-id", 12345, "test-model");
    assert_eq!(openai_chunk.id, "test-id");
    assert_eq!(openai_chunk.created, 12345);
    assert_eq!(openai_chunk.model, "test-model");
    assert_eq!(openai_chunk.choices.len(), 1);
    assert_eq!(
        openai_chunk.choices[0].delta.content,
        Some("Hello".to_string())
    );
    assert_eq!(openai_chunk.choices[0].finish_reason, None);
}

#[test]
fn test_openai_to_anthropic_conversion() {
    let openai_req = OpenAIChatRequest {
        model: "claude-3-opus-20240229".to_string(),
        messages: vec![
            openai_message("system", "You are a helpful assistant."),
            openai_message("user", "Hello!"),
        ],
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(1000),
        stop: Some(vec!["<stop>".to_string()]),
        stream: Some(true),
        presence_penalty: None,
        frequency_penalty: None,
        ..Default::default()
    };

    let anthropic_req = convert_openai_to_anthropic(&openai_req);
    assert_eq!(anthropic_req.model, "claude-3-opus-20240229");
    assert_eq!(
        anthropic_req.system,
        Some("You are a helpful assistant.".to_string())
    );
    assert_eq!(anthropic_req.messages.len(), 1);
    assert_eq!(anthropic_req.messages[0].role, "user");
    assert_eq!(anthropic_req.messages[0].content, "Hello!");
    assert_eq!(anthropic_req.temperature, Some(0.7));
    assert_eq!(anthropic_req.top_p, Some(0.9));
    assert_eq!(anthropic_req.max_tokens, 1000);
    assert_eq!(
        anthropic_req.stop_sequences,
        Some(vec!["<stop>".to_string()])
    );
    assert_eq!(anthropic_req.stream, Some(true));
}

#[test]
fn test_openai_to_anthropic_no_system_message() {
    let openai_req = OpenAIChatRequest {
        model: "claude-3-sonnet".to_string(),
        messages: vec![openai_message("user", "Hello!")],
        max_tokens: Some(500),
        temperature: None,
        top_p: None,
        stop: None,
        stream: None,
        presence_penalty: None,
        frequency_penalty: None,
        ..Default::default()
    };

    let anthropic_req = convert_openai_to_anthropic(&openai_req);
    assert_eq!(anthropic_req.model, "claude-3-sonnet");
    assert!(anthropic_req.system.is_none());
    assert_eq!(anthropic_req.messages.len(), 1);
    assert_eq!(anthropic_req.max_tokens, 500);
}

#[test]
fn test_anthropic_to_openai_response_conversion() {
    let anthropic_resp = AnthropicChatResponse {
        id: "msg_123".to_string(),
        model: "claude-3-opus-20240229".to_string(),
        content: vec![AnthropicContent {
            content_type: "text".to_string(),
            text: Some("Hello! How can I help you today?".to_string()),
        }],
        usage: AnthropicUsage {
            input_tokens: 10,
            output_tokens: 20,
        },
        stop_reason: Some("end_turn".to_string()),
    };

    let openai_resp = convert_anthropic_to_openai(&anthropic_resp, "test-model");
    assert_eq!(openai_resp.id, "msg_123");
    assert_eq!(openai_resp.model, "test-model");
    assert_eq!(openai_resp.choices.len(), 1);
    assert!(openai_resp.choices[0].message.is_some());
    let message = openai_resp.choices[0].message.as_ref().unwrap();
    assert_eq!(message.role, "assistant");
    assert_eq!(
        message.content.as_ref().map(|c| c.as_string()).as_deref(),
        Some("Hello! How can I help you today?")
    );
    assert_eq!(openai_resp.usage.prompt_tokens, 10);
    assert_eq!(openai_resp.usage.completion_tokens, 20);
    assert_eq!(openai_resp.usage.total_tokens, 30);
    assert_eq!(
        openai_resp.choices[0].finish_reason,
        Some("stop".to_string())
    );
}

#[test]
fn test_anthropic_stream_chunk_to_openai() {
    let anthropic_chunk = AnthropicStreamChunk {
        id: "msg_123".to_string(),
        model: "claude-3-opus".to_string(),
        delta: Some(AnthropicDelta {
            text: Some("Hello".to_string()),
            stop_reason: None,
        }),
        usage: None,
        stop_reason: None,
    };

    let openai_chunk =
        convert_anthropic_stream_chunk_to_openai(&anthropic_chunk, 12345, "test-model");
    assert_eq!(openai_chunk.id, "msg_123");
    assert_eq!(openai_chunk.created, 12345);
    assert_eq!(openai_chunk.model, "test-model");
    assert_eq!(openai_chunk.choices.len(), 1);
    assert_eq!(
        openai_chunk.choices[0].delta.content,
        Some("Hello".to_string())
    );
}

#[test]
fn test_openai_to_gemini_conversion() {
    let openai_req = OpenAIChatRequest {
        model: "gemini-pro".to_string(),
        messages: vec![
            openai_message("user", "Hello!"),
            openai_message("assistant", "Hi there! How can I help?"),
            openai_message("user", "Tell me a joke."),
        ],
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(100),
        stop: Some(vec!["\n".to_string()]),
        presence_penalty: None,
        frequency_penalty: None,
        stream: None,
        ..Default::default()
    };

    let gemini_req = convert_openai_to_gemini(&openai_req);
    assert_eq!(gemini_req.contents.len(), 3);
    assert_eq!(gemini_req.contents[0].role, "user");
    assert_eq!(
        gemini_req.contents[0].parts[0].text,
        Some("Hello!".to_string())
    );
    assert_eq!(gemini_req.contents[1].role, "model");
    assert_eq!(
        gemini_req.contents[1].parts[0].text,
        Some("Hi there! How can I help?".to_string())
    );
    assert_eq!(gemini_req.contents[2].role, "user");
    assert_eq!(
        gemini_req.contents[2].parts[0].text,
        Some("Tell me a joke.".to_string())
    );

    assert!(gemini_req.generation_config.is_some());
    let gen_config = gemini_req.generation_config.as_ref().unwrap();
    assert_eq!(gen_config.temperature, Some(0.7));
    assert_eq!(gen_config.top_p, Some(0.9));
    assert_eq!(gen_config.max_output_tokens, Some(100));
    assert_eq!(gen_config.stop_sequences, Some(vec!["\n".to_string()]));
}

#[test]
fn test_gemini_to_openai_response_conversion() {
    let gemini_resp = GeminiChatResponse {
        candidates: vec![GeminiCandidate {
            content: GeminiContent {
                role: "model".to_string(),
                parts: vec![GeminiPart {
                    text: Some(
                        "Why did the chicken cross the road? To get to the other side!".to_string(),
                    ),
                    function_response: None,
                }],
            },
            finish_reason: Some("STOP".to_string()),
        }],
        usage_metadata: Some(GeminiUsageMetadata {
            prompt_token_count: 10,
            candidates_token_count: 20,
            total_token_count: 30,
        }),
    };

    let openai_resp = convert_gemini_to_openai(&gemini_resp, "gemini-pro", 12345);
    assert_eq!(openai_resp.choices.len(), 1);
    assert!(openai_resp.choices[0].message.is_some());
    let message = openai_resp.choices[0].message.as_ref().unwrap();
    assert_eq!(message.role, "assistant");
    assert_eq!(
        message.content.as_ref().map(|c| c.as_string()).as_deref(),
        Some("Why did the chicken cross the road? To get to the other side!")
    );
    assert_eq!(openai_resp.usage.prompt_tokens, 10);
    assert_eq!(openai_resp.usage.completion_tokens, 20);
    assert_eq!(openai_resp.usage.total_tokens, 30);
    assert_eq!(
        openai_resp.choices[0].finish_reason,
        Some("stop".to_string())
    );
}

#[test]
fn test_gemini_stream_chunk_to_openai() {
    let gemini_chunk = GeminiStreamChunk {
        candidates: vec![GeminiCandidate {
            content: GeminiContent {
                role: "model".to_string(),
                parts: vec![GeminiPart {
                    text: Some("Hello".to_string()),
                    function_response: None,
                }],
            },
            finish_reason: None,
        }],
        usage_metadata: None,
    };

    let openai_chunk =
        convert_gemini_stream_chunk_to_openai(&gemini_chunk, "test-id", 12345, "gemini-pro");
    assert_eq!(openai_chunk.id, "test-id");
    assert_eq!(openai_chunk.created, 12345);
    assert_eq!(openai_chunk.model, "gemini-pro");
    assert_eq!(openai_chunk.choices.len(), 1);
    assert_eq!(
        openai_chunk.choices[0].delta.content,
        Some("Hello".to_string())
    );
}

#[test]
fn test_convert_openai_message_with_tool_calls_to_ollama() {
    let req = OpenAIChatRequest {
        model: "llama3".to_string(),
        messages: vec![
            OpenAIMessage {
                role: "user".to_string(),
                content: Some(MessageContent::String("What's the weather?".to_string())),
                ..Default::default()
            },
            OpenAIMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![OpenAIToolCall {
                    id: Some("call_123".to_string()),
                    r#type: Some("function".to_string()),
                    function: Some(OpenAIToolCallFunction {
                        name: Some("get_weather".to_string()),
                        arguments: Some(r#"{"location":"Beijing"}"#.to_string()),
                    }),
                    index: Some(0),
                }]),
                ..Default::default()
            },
            OpenAIMessage {
                role: "tool".to_string(),
                content: Some(MessageContent::String(
                    r#"{"temperature":"25°C"}"#.to_string(),
                )),
                tool_call_id: Some("call_123".to_string()),
                name: Some("get_weather".to_string()),
                ..Default::default()
            },
        ],
        tools: Some(vec![OpenAITool {
            id: Some("weather_tool".to_string()),
            r#type: "function".to_string(),
            function: Some(OpenAIToolFunction {
                name: "get_weather".to_string(),
                strict: None,
                description: Some("Get weather for a location".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                })),
            }),
        }]),
        ..Default::default()
    };

    let ollama_req = convert_openai_to_ollama(&req);
    // Verify tools are converted
    assert!(ollama_req.tools.is_some());
}

#[test]
fn test_convert_openai_message_with_tool_to_anthropic() {
    use lumina::convert::convert_openai_to_anthropic;
    use lumina::types::*;

    let req = OpenAIChatRequest {
        model: "claude-3-5-sonnet".to_string(),
        messages: vec![OpenAIMessage {
            role: "user".to_string(),
            content: Some(MessageContent::String("What's the weather?".to_string())),
            ..Default::default()
        }],
        tools: Some(vec![OpenAITool {
            id: Some("weather_tool".to_string()),
            r#type: "function".to_string(),
            function: Some(OpenAIToolFunction {
                name: "get_weather".to_string(),
                strict: None,
                description: Some("Get weather for a location".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                })),
            }),
        }]),
        ..Default::default()
    };

    let anthropic_req = convert_openai_to_anthropic(&req);
    // Anthropic has tools at request level
    assert!(anthropic_req.tools.is_some());
}

#[test]
fn test_convert_openai_message_with_tool_to_gemini() {
    use lumina::convert::convert_openai_to_gemini;
    use lumina::types::*;

    let req = OpenAIChatRequest {
        model: "gemini-pro".to_string(),
        messages: vec![OpenAIMessage {
            role: "user".to_string(),
            content: Some(MessageContent::String("What's the weather?".to_string())),
            ..Default::default()
        }],
        tools: Some(vec![OpenAITool {
            id: Some("weather_tool".to_string()),
            r#type: "function".to_string(),
            function: Some(OpenAIToolFunction {
                name: "get_weather".to_string(),
                strict: None,
                description: Some("Get weather for a location".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                })),
            }),
        }]),
        ..Default::default()
    };

    let gemini_req = convert_openai_to_gemini(&req);
    // Gemini has tools in contents or generation_config
    assert!(gemini_req.tools.is_some());
}

// =============================================================================
// Responses API Conversion Tests
// =============================================================================

#[test]
fn test_convert_responses_to_chat_with_string_input() {
    use lumina::convert::convert_responses_to_chat;
    use lumina::types::*;

    let responses_req = ResponsesRequest {
        model: "gpt-4o".to_string(),
        input: Some(ResponseInput::String("Hello, how are you?".to_string())),
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_output_tokens: Some(100),
        stream: Some(false),
        ..Default::default()
    };

    let chat_req = convert_responses_to_chat(&responses_req);
    assert_eq!(chat_req.model, "gpt-4o");
    assert_eq!(chat_req.messages.len(), 1);
    assert_eq!(chat_req.messages[0].role, "user");
    assert_eq!(
        chat_req.messages[0]
            .content
            .as_ref()
            .map(|c| c.as_string())
            .as_deref(),
        Some("Hello, how are you?")
    );
    assert_eq!(chat_req.temperature, Some(0.7));
    assert_eq!(chat_req.top_p, Some(0.9));
    assert_eq!(chat_req.max_tokens, Some(100));
    assert_eq!(chat_req.stream, Some(false));
}

#[test]
fn test_convert_responses_to_chat_with_message_array() {
    use lumina::convert::convert_responses_to_chat;
    use lumina::types::*;

    let responses_req = ResponsesRequest {
        model: "gpt-4o".to_string(),
        input: Some(ResponseInput::Messages(vec![
            OpenAIMessage {
                role: "system".to_string(),
                content: Some(MessageContent::String(
                    "You are a helpful assistant".to_string(),
                )),
                ..Default::default()
            },
            OpenAIMessage {
                role: "user".to_string(),
                content: Some(MessageContent::String("Hello!".to_string())),
                ..Default::default()
            },
        ])),
        ..Default::default()
    };

    let chat_req = convert_responses_to_chat(&responses_req);
    assert_eq!(chat_req.model, "gpt-4o");
    assert_eq!(chat_req.messages.len(), 2);
    assert_eq!(chat_req.messages[0].role, "system");
    assert_eq!(chat_req.messages[1].role, "user");
    assert_eq!(
        chat_req.messages[1]
            .content
            .as_ref()
            .map(|c| c.as_string())
            .as_deref(),
        Some("Hello!")
    );
}

#[test]
fn test_convert_responses_to_chat_with_raw_json_input() {
    use lumina::convert::convert_responses_to_chat;
    use lumina::types::*;
    use serde_json::json;

    // Test with complex raw JSON input (array format)
    let responses_req = ResponsesRequest {
        model: "gpt-4o".to_string(),
        input: Some(ResponseInput::Raw(json!([
            {"role": "user", "content": "Hello from raw JSON!"}
        ]))),
        ..Default::default()
    };

    let chat_req = convert_responses_to_chat(&responses_req);
    assert_eq!(chat_req.model, "gpt-4o");
    assert_eq!(chat_req.messages.len(), 1);
    assert_eq!(chat_req.messages[0].role, "user");
}

#[test]
fn test_convert_responses_to_chat_with_instructions() {
    use lumina::convert::convert_responses_to_chat;
    use lumina::types::*;

    // Test with instructions (should become system message) and string input
    let responses_req = ResponsesRequest {
        model: "gpt-4o".to_string(),
        instructions: Some("You are a helpful coding assistant".to_string()),
        input: Some(ResponseInput::String("Hello, how are you?".to_string())),
        ..Default::default()
    };

    let chat_req = convert_responses_to_chat(&responses_req);
    assert_eq!(chat_req.model, "gpt-4o");
    assert_eq!(chat_req.messages.len(), 2);
    // Instructions should be first message as system role
    assert_eq!(chat_req.messages[0].role, "system");
    assert_eq!(
        chat_req.messages[0]
            .content
            .as_ref()
            .map(|c| c.as_string())
            .as_deref(),
        Some("You are a helpful coding assistant")
    );
    // Input should be second message as user role
    assert_eq!(chat_req.messages[1].role, "user");
    assert_eq!(
        chat_req.messages[1]
            .content
            .as_ref()
            .map(|c| c.as_string())
            .as_deref(),
        Some("Hello, how are you?")
    );
}

#[test]
fn test_convert_responses_to_chat_with_instructions_only() {
    use lumina::convert::convert_responses_to_chat;
    use lumina::types::*;

    // Test with instructions only (no input)
    let responses_req = ResponsesRequest {
        model: "gpt-4o".to_string(),
        instructions: Some("You are a helpful assistant".to_string()),
        input: None,
        ..Default::default()
    };

    let chat_req = convert_responses_to_chat(&responses_req);
    assert_eq!(chat_req.model, "gpt-4o");
    assert_eq!(chat_req.messages.len(), 1);
    assert_eq!(chat_req.messages[0].role, "system");
    assert_eq!(
        chat_req.messages[0]
            .content
            .as_ref()
            .map(|c| c.as_string())
            .as_deref(),
        Some("You are a helpful assistant")
    );
}

#[test]
fn test_convert_chat_to_responses() {
    use lumina::convert::convert_chat_to_responses;
    use lumina::types::*;

    let chat_resp = OpenAIChatResponse {
        id: "chatcmpl-123".to_string(),
        object: "chat.completion".to_string(),
        created: 1620000000,
        model: "gpt-4o".to_string(),
        choices: vec![OpenAIChoice {
            index: 0,
            message: Some(OpenAIMessage {
                role: "assistant".to_string(),
                content: Some(MessageContent::String("Hello there!".to_string())),
                ..Default::default()
            }),
            delta: None,
            finish_reason: Some("stop".to_string()),
        }],
        usage: OpenAIUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        },
    };

    let responses_resp = convert_chat_to_responses(&chat_resp, 1620000000);
    assert_eq!(responses_resp.id, "chatcmpl-123");
    assert_eq!(responses_resp.object, "response");
    assert_eq!(responses_resp.created_at, 1620000000);
    assert_eq!(responses_resp.model, "gpt-4o");
    assert_eq!(responses_resp.status, "completed");
    assert_eq!(responses_resp.output.len(), 1);
    assert_eq!(responses_resp.output[0].role, Some("assistant".to_string()));

    let content = &responses_resp.output[0].content.as_ref().unwrap()[0];
    assert_eq!(content.text, Some("Hello there!".to_string()));

    let usage = responses_resp.usage.unwrap();
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 20);
    assert_eq!(usage.total_tokens, 30);
}

#[test]
fn test_create_response_created_event() {
    use lumina::convert::create_response_created_event;

    let (event_type, json) = create_response_created_event("resp_123", "gpt-4o", 1620000000);
    assert_eq!(event_type, "response.created");
    assert_eq!(json["response"]["id"], "resp_123");
    assert_eq!(json["response"]["model"], "gpt-4o");
    assert_eq!(json["response"]["status"], "in_progress");
}

#[test]
fn test_responses_request_serialization() {
    use lumina::types::*;

    let req = ResponsesRequest {
        model: "gpt-4o".to_string(),
        input: Some(ResponseInput::String("Hello".to_string())),
        temperature: Some(0.7),
        stream: Some(true),
        ..Default::default()
    };

    let json_str = serde_json::to_string(&req).unwrap();
    let deserialized: ResponsesRequest = serde_json::from_str(&json_str).unwrap();

    assert_eq!(deserialized.model, "gpt-4o");
    assert!(matches!(deserialized.input, Some(ResponseInput::String(_))));
    assert_eq!(deserialized.temperature, Some(0.7));
    assert_eq!(deserialized.stream, Some(true));
}

#[test]
fn test_responses_response_serialization() {
    use lumina::types::*;

    let resp = ResponsesResponse {
        id: "resp_123".to_string(),
        object: "response".to_string(),
        created_at: 1620000000,
        model: "gpt-4o".to_string(),
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
                text: Some("Hello!".to_string()),
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
        status: "completed".to_string(),
        text: None,
        truncation: None,
        usage: Some(ResponseUsage {
            input_tokens: 10,
            input_tokens_details: None,
            output_tokens: 5,
            output_tokens_details: None,
            total_tokens: 15,
        }),
        user: None,
        store: None,
    };

    let json_str = serde_json::to_string(&resp).unwrap();
    let deserialized: ResponsesResponse = serde_json::from_str(&json_str).unwrap();

    assert_eq!(deserialized.id, "resp_123");
    assert_eq!(deserialized.object, "response");
    assert_eq!(deserialized.status, "completed");
    assert_eq!(deserialized.usage.unwrap().total_tokens, 15);
}

// =============================================================================
