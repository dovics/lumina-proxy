//! Common type definitions for LLM requests and responses

use serde::{Deserialize, Serialize};
use thiserror::Error;

// =============================================================================
// OpenAI - Standard API Types (used by all incoming requests)
// =============================================================================

/// OpenAI chat completion request (standard format for all incoming requests)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,
}

/// A single message in an OpenAI chat conversation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<OpenAIFunctionCall>, // legacy format
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    pub tool_call_id: Option<String>,
}

/// OpenAI chat completion response (standard format for all outgoing responses)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: OpenAIUsage,
}

/// A single choice in an OpenAI chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIChoice {
    pub index: u32,
    pub message: Option<OpenAIMessage>,
    pub delta: Option<OpenAIDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Usage statistics for an OpenAI request
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Delta content for streaming responses
/// Tool call function in delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Tool call in delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<OpenAIToolCallFunction>,
}

/// Function call - legacy format used in some API versions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Function specification for a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Tool definition in request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAITool {
    pub id: Option<String>,
    pub r#type: String, // always "function"
    pub function: OpenAIToolFunction,
}

/// Tool choice - can be string or object
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIToolChoice {
    String(String), // "auto", "none", "required"
    Object {
        #[serde(rename = "type")]
        type_: String, // "function"
        function: OpenAIToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

/// Streaming chunk for OpenAI format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<OpenAIStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

/// Streaming choice for OpenAI chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStreamChoice {
    pub index: u32,
    pub delta: OpenAIDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// OpenAI model list entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModel {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<u32>,
}

/// OpenAI /v1/models list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIModelsListResponse {
    pub object: String,
    pub data: Vec<OpenAIModel>,
}

// =============================================================================
// Ollama - Native Types
// =============================================================================

/// Ollama chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OllamaTool>>,
}

/// A single message in an Ollama chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

/// Ollama tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaTool {
    pub type_: String, // "function"
    pub function: OllamaToolFunction,
}

/// Ollama tool function specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Ollama chat completion response (non-streaming)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: OllamaMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
}

/// Ollama streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaStreamChunk {
    pub model: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<OllamaDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<OllamaMessage>,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
}

/// Delta content for Ollama streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaDelta {
    pub content: String,
}

/// Ollama function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionCall {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Ollama tool call in message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    pub function: OllamaFunctionCall,
}

// =============================================================================
// Anthropic - Native Types
// =============================================================================

/// Anthropic Messages API request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicChatRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "stop_sequences")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
}

/// A single message in an Anthropic conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

/// Anthropic Messages API response (non-streaming)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicChatResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<AnthropicContent>,
    pub usage: AnthropicUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

/// Content block in Anthropic response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicContent {
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Token usage statistics for Anthropic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Anthropic streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicStreamChunk {
    pub id: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<AnthropicDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<AnthropicUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

/// Delta content for Anthropic streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

/// Anthropic tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

// =============================================================================
// Gemini - Native Types
// =============================================================================

/// Gemini function declaration for tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionDeclaration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Gemini tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiTool {
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

/// Gemini function response for tool results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: String,
}

/// Gemini generate content request (for chat)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiChatRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
}

/// Content block for Gemini
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

/// Content part for Gemini
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<GeminiFunctionResponse>,
}

/// Generation configuration for Gemini
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "top_p")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "stopSequences")]
    pub stop_sequences: Option<Vec<String>>,
}

/// Gemini chat completion response (non-streaming)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiChatResponse {
    pub candidates: Vec<GeminiCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

/// Gemini candidate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiCandidate {
    pub content: GeminiContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Usage metadata for Gemini
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiUsageMetadata {
    pub prompt_token_count: u32,
    pub candidates_token_count: u32,
    pub total_token_count: u32,
}

/// Gemini streaming response chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiStreamChunk {
    pub candidates: Vec<GeminiCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

// =============================================================================
// Error Type
// =============================================================================

/// Proxy error type
#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Routing error: {0}")]
    RoutingError(String),

    #[error("Format conversion error: {0}")]
    ConversionError(String),

    #[error("Backend request error: {0}")]
    BackendRequestError(String),

    #[error("Token counting error: {0}")]
    TokenCountError(String),
}

// =============================================================================
// OpenAI Responses API Types
// =============================================================================

/// Response input content - can be a simple string or an array of messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponseInput {
    String(String),
    Messages(Vec<OpenAIMessage>),
}

impl Default for ResponseInput {
    fn default() -> Self {
        ResponseInput::String(String::new())
    }
}

/// Response usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InputTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
    #[serde(default)]
    pub cached_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_tokens: Option<u32>,
}

/// Content part in response output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseContentPart {
    #[serde(rename = "type")]
    pub content_type: String, // "output_text"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<serde_json::Value>>,
}

/// Output item in response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseOutputItem {
    #[serde(rename = "type")]
    pub output_type: String, // "message"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<ResponseContentPart>>,
}

/// Reasoning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<bool>,
}

/// Text configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
}

/// Incomplete details for failed responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncompleteDetails {
    pub reason: String,
}

/// OpenAI Responses API request
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<ResponseInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<serde_json::Value>,
}

/// OpenAI Responses API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub created_at: i64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<IncompleteDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    pub output: Vec<ResponseOutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAIToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<serde_json::Value>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

// =============================================================================
// Token Statistics
// =============================================================================

/// Statistics for a completed request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestStatistics {
    pub timestamp: String,
    pub model: String,
    pub provider: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub duration_ms: u64,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}
