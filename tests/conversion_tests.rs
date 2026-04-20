use lumina_proxy::convert::{
    convert_openai_to_ollama, convert_ollama_to_openai,
    convert_ollama_stream_chunk_to_openai,
    convert_openai_to_anthropic, convert_anthropic_to_openai,
    convert_anthropic_stream_chunk_to_openai,
    convert_openai_to_gemini, convert_gemini_to_openai,
    convert_gemini_stream_chunk_to_openai,
};
use lumina_proxy::types::{
    OpenAIChatRequest, OpenAIMessage,
    OllamaChatResponse, OllamaStreamChunk, OllamaDelta, OllamaMessage,
    AnthropicChatResponse, AnthropicStreamChunk, AnthropicDelta, AnthropicContent, AnthropicUsage,
    GeminiChatResponse, GeminiStreamChunk, GeminiCandidate, GeminiContent, GeminiPart,
    GeminiUsageMetadata,
};

fn openai_message(role: &str, content: &str) -> OpenAIMessage {
    OpenAIMessage {
        role: role.to_string(),
        content: content.to_string(),
    }
}

#[test]
fn test_openai_to_ollama_conversion() {
    let openai_req = OpenAIChatRequest {
        model: "llama3:8b".to_string(),
        messages: vec![
            openai_message("user", "Hello!"),
        ],
        temperature: Some(0.7),
        top_p: Some(0.9),
        max_tokens: Some(100),
        stream: Some(true),
        stop: Some(vec!["<stop>".to_string()]),
        presence_penalty: Some(0.1),
        frequency_penalty: Some(0.2),
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
    assert_eq!(message.content, "Hello! How can I help you today?");
    assert_eq!(openai_resp.usage.prompt_tokens, 5);
    assert_eq!(openai_resp.usage.completion_tokens, 20);
    assert_eq!(openai_resp.usage.total_tokens, 25);
    assert_eq!(openai_resp.choices[0].finish_reason, Some("stop".to_string()));
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

    let openai_chunk = convert_ollama_stream_chunk_to_openai(&ollama_chunk, "test-id", 12345, "test-model");
    assert_eq!(openai_chunk.id, "test-id");
    assert_eq!(openai_chunk.created, 12345);
    assert_eq!(openai_chunk.model, "test-model");
    assert_eq!(openai_chunk.choices.len(), 1);
    assert_eq!(openai_chunk.choices[0].delta.content, Some("Hello".to_string()));
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
    };

    let anthropic_req = convert_openai_to_anthropic(&openai_req);
    assert_eq!(anthropic_req.model, "claude-3-opus-20240229");
    assert_eq!(anthropic_req.system, Some("You are a helpful assistant.".to_string()));
    assert_eq!(anthropic_req.messages.len(), 1);
    assert_eq!(anthropic_req.messages[0].role, "user");
    assert_eq!(anthropic_req.messages[0].content, "Hello!");
    assert_eq!(anthropic_req.temperature, Some(0.7));
    assert_eq!(anthropic_req.top_p, Some(0.9));
    assert_eq!(anthropic_req.max_tokens, 1000);
    assert_eq!(anthropic_req.stop_sequences, Some(vec!["<stop>".to_string()]));
    assert_eq!(anthropic_req.stream, Some(true));
}

#[test]
fn test_openai_to_anthropic_no_system_message() {
    let openai_req = OpenAIChatRequest {
        model: "claude-3-sonnet".to_string(),
        messages: vec![
            openai_message("user", "Hello!"),
        ],
        max_tokens: Some(500),
        temperature: None,
        top_p: None,
        stop: None,
        stream: None,
        presence_penalty: None,
        frequency_penalty: None,
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
        content: vec![
            AnthropicContent {
                content_type: "text".to_string(),
                text: Some("Hello! How can I help you today?".to_string()),
            }
        ],
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
    assert_eq!(message.content, "Hello! How can I help you today?");
    assert_eq!(openai_resp.usage.prompt_tokens, 10);
    assert_eq!(openai_resp.usage.completion_tokens, 20);
    assert_eq!(openai_resp.usage.total_tokens, 30);
    assert_eq!(openai_resp.choices[0].finish_reason, Some("stop".to_string()));
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

    let openai_chunk = convert_anthropic_stream_chunk_to_openai(&anthropic_chunk, 12345, "test-model");
    assert_eq!(openai_chunk.id, "msg_123");
    assert_eq!(openai_chunk.created, 12345);
    assert_eq!(openai_chunk.model, "test-model");
    assert_eq!(openai_chunk.choices.len(), 1);
    assert_eq!(openai_chunk.choices[0].delta.content, Some("Hello".to_string()));
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
    };

    let gemini_req = convert_openai_to_gemini(&openai_req);
    assert_eq!(gemini_req.contents.len(), 3);
    assert_eq!(gemini_req.contents[0].role, "user");
    assert_eq!(gemini_req.contents[0].parts[0].text, Some("Hello!".to_string()));
    assert_eq!(gemini_req.contents[1].role, "model");
    assert_eq!(gemini_req.contents[1].parts[0].text, Some("Hi there! How can I help?".to_string()));
    assert_eq!(gemini_req.contents[2].role, "user");
    assert_eq!(gemini_req.contents[2].parts[0].text, Some("Tell me a joke.".to_string()));

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
        candidates: vec![
            GeminiCandidate {
                content: GeminiContent {
                    role: "model".to_string(),
                    parts: vec![
                        GeminiPart {
                            text: Some("Why did the chicken cross the road? To get to the other side!".to_string()),
                        }
                    ],
                },
                finish_reason: Some("STOP".to_string()),
            }
        ],
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
    assert_eq!(message.content, "Why did the chicken cross the road? To get to the other side!");
    assert_eq!(openai_resp.usage.prompt_tokens, 10);
    assert_eq!(openai_resp.usage.completion_tokens, 20);
    assert_eq!(openai_resp.usage.total_tokens, 30);
    assert_eq!(openai_resp.choices[0].finish_reason, Some("stop".to_string()));
}

#[test]
fn test_gemini_stream_chunk_to_openai() {
    let gemini_chunk = GeminiStreamChunk {
        candidates: vec![
            GeminiCandidate {
                content: GeminiContent {
                    role: "model".to_string(),
                    parts: vec![
                        GeminiPart {
                            text: Some("Hello".to_string()),
                        }
                    ],
                },
                finish_reason: None,
            }
        ],
        usage_metadata: None,
    };

    let openai_chunk = convert_gemini_stream_chunk_to_openai(&gemini_chunk, "test-id", 12345, "gemini-pro");
    assert_eq!(openai_chunk.id, "test-id");
    assert_eq!(openai_chunk.created, 12345);
    assert_eq!(openai_chunk.model, "gemini-pro");
    assert_eq!(openai_chunk.choices.len(), 1);
    assert_eq!(openai_chunk.choices[0].delta.content, Some("Hello".to_string()));
}
